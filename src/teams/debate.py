import asyncio
import pandas as pd
import json
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

from agents.linguist import LinguistAgent

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# AutoGen 0.4+ Imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

class LinguistTurn(BaseModel):
    """The structured output for a linguist's turn in the debate."""
    agent_name: str = Field(description="The name of the linguist speaking.")
    argument: str = Field(description="Your critique. Reference specific peers if agreeing/disagreeing. If this is a closing statement, include all specific details as to justify your score, even if the idea originated from a peer. This field should be a string containing Markdown formatted text.")
    proposed_score: int = Field(description="The score (1-10) you currently advocate for.")

class ModeratorTurn(BaseModel):
    """The output from the moderator."""
    intervene: bool = Field(description="Whether the moderator is stepping in to intervene. False if there is no need to intervene.")
    violators: List[str] = Field(description="The names of the participants who require intervention. Empty if there is no need to intervene.")
    feedback: str = Field(description="The feedback to give the debate participant if there is an intervention. Empty if there is no need to intervene.")

class DebateCommentator(BaseModel):
    """The final consensus output from the moderator."""
    chapter: int = Field(description="The chapter debated.")
    verse: str = Field(description="The verse debated.")
    final_consensus_score: int = Field(description="The final agreed-upon score (integer). Choose the lowest score in the event of non-consensus.")
    consensus_summary: str = Field(description="A detailed explanation of why this score was chosen.")
    closing_statements: List[str] = Field(description="Closing remarks from the participants in markdown, beginning with a section heading of the participant name.")

task_description = f"""
    You are a participant in a translation debate over how to score a translation of Greek text. 
    You will debate the other participants and try to come to a consensus as to a score. 
    Be critical and err on the side of a lower score.
"""

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "linguist_review",
        "schema": LinguistTurn.model_json_schema()
    }
}

class Debate:
    def __init__(self):
        # self.linguists = [
        #     LinguistAgent("GEMINI_LINGUIST", "gemini-3-pro-preview", GEMINI_KEY, GOOGLE_BASE_URL, task_description, response_format),
        #     LinguistAgent("GPT5_LINGUIST", "gpt-5", OPENAI_KEY, None, task_description, response_format)
        # ]

        self.linguists = [
            LinguistAgent("GEMINI_LINGUIST", "gemini-2.0-flash-lite", GEMINI_KEY, GOOGLE_BASE_URL, task_description, response_format),
            LinguistAgent("GPT5_LINGUIST", "gpt-4o", OPENAI_KEY, None, task_description, response_format)
        ]

        

    async def run_single_verse_debate(self, group_df: pd.DataFrame):
        """
        Runs a RoundRobin debate for a single group of 3 dataframe rows (one verse).
        """
        # Extract metadata from the first row of the group
        chapter = group_df.iloc[0]['Chapter']
        verse = group_df.iloc[0]['Verse']
        face = group_df.iloc[0]['Face_Annotation']
        greek_text = group_df.iloc[0]['Greek_Text']
        translation = group_df.iloc[0]['Translation']
        
        print(f"\n--- 🗣️  Initiating Debate for {chapter} {verse} ---")

        # 1. Construct the Context from the DataFrame Rows
        # We map the specific agent names to their previous independent analysis
        initial_context = f"## Debate Context for {chapter} {verse}\n"
        initial_context = f"**Greek Text** {greek_text}\n\n"
        initial_context = f"** Translation ** {translation}\n\n"
        initial_context += f"**Face Annotation:** {face}\n\n"
        initial_context += "### Initial Independent Analyses:\n"
        
        for _, row in group_df.iterrows():
            initial_context += (
                f"- **{row['Agent_Name']}** (Initial Score: {row['Score']}): {row['Model_Analysis']}\n"
            )
        
        initial_context += (
            "\n**TASK:** Debate these initial findings. Critique each other. "
            "Come to a consensus score. Err on the side of being critical (lower scores)."
        )
        

        # Moderator must output the final JSON summary
        moderator_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            response_format=ModeratorTurn,
            key=OPENAI_KEY
        )

        moderator = AssistantAgent(
            name="Moderator", 
            model_client=moderator_client, 
            system_message=f"""
            You are a moderator of a debate between linguists. 
            The linguists are supposed to be discussing the translation of the following Greek text: 
            >>> {greek_text}

            The translation being evaluated is as follows:
            >>> {translation}

            The translation is intended to retain the following:
            >>> {face}

            Listen to the debate. Make sure the participants only evaluate the translation from a linguistic perspective.
            
            ONLY Intervene as a moderator if a linguist's response **DOES NOT** meet the following criteria:
            1. The analysis must be based on verifiable linguistic, stylistic, or semantic arguments.
            2. Words and phrases being analyzed **MUST** be present in the texts being analyzed.
            """
        )

        # 4. Define Team (RoundRobin)
        debators = [x.get_agent() for x in self.linguists]
        
        rounds = 2
        debate_rounds =  (len(debators) + 1) * rounds + 1
        termination = MaxMessageTermination(max_messages=debate_rounds)

        debate_team = RoundRobinGroupChat(
            participants=[*debators, moderator],
            termination_condition=termination
        )

        debate_results = await debate_team.run(task=initial_context)

        await termination.reset()
        termination._max_messages = len(debators) + 1

        print("--- Debate ---")
        debate = []
        for message in debate_results.messages[1:]:
            # print(message)
            debate.append(message.content)

        closing_results = await debate_team.run(task="""
            Now we transition to closing statements. 
            Participants, please give your closing statement providing all details to justify your final score.
        """)

        print("--- Closing Statements ---")
        closing_statements = []
        for message in closing_results.messages:
            # print(message)
            closing_statements.append(message.content)

        row_data = {
            'Chapter': chapter,
            'Verse': verse,
            'Face_Annotation': face,
            'Greek_Text': greek_text,
            'Translation': translation,
            "Debate": json.dumps(debate),
            "Closing_Statements": json.dumps(closing_statements)
        }
        
        return pd.DataFrame([row_data])

    async def process_interleaved_dataframe(self, df: pd.DataFrame):
        final_results = []

        # GROUP BY: This automatically handles the "3 rows at a time" requirement 
        # regardless of whether the rows are perfectly sorted or mixed up.
        grouped_data = df.groupby(['Chapter', 'Verse'])

        for (chapter, verse), group_df in grouped_data:
            
            # Validation: Ensure we actually have 3 agents for this verse
            if len(group_df) != 2:
                print(f"⚠️ Skipping {chapter} {verse}: Expected 2 analyses, found {len(group_df)}")
                continue

            # Run the async debate for this specific group
            consensus_data = await self.run_single_verse_debate(group_df)
            print(consensus_data)
            final_results.append(consensus_data)

        print("---------- FINAL RESULTS -----------")
        print(final_results)

        return pd.concat(final_results)
    
async def run(df):
    debate = Debate()
    results = await debate.process_interleaved_dataframe(df)
    print(results)

if __name__ == "__main__":
    df = pd.read_csv("debate_analysis_results.csv")
    asyncio(run(df))