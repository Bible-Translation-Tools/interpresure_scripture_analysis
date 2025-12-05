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
    argument: str = Field(description="Your critique. Reference specific peers if agreeing/disagreeing.")
    proposed_score: int = Field(description="The score (1-10) you currently advocate for.")

class ModeratorConsensus(BaseModel):
    """The final consensus output from the moderator."""
    chapter: str = Field(description="The chapter debated.")
    verse: str = Field(description="The verse debated.")
    final_consensus_score: int = Field(description="The final agreed-upon score (integer). Choose the lowest score in the event of non-consensus.")
    consensus_summary: str = Field(description="A concise summary of why this score was chosen.")
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
        self.linguists = [
            LinguistAgent("GEMINI_LINGUIST", "gemini-3-pro-preview", GEMINI_KEY, GOOGLE_BASE_URL, task_description, response_format),
            LinguistAgent("GPT5_LINGUIST", "gpt-5", OPENAI_KEY, None, task_description, response_format)
        ]

    async def run_single_verse_debate(self, group_df: pd.DataFrame) -> dict:
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
            "\n**TASK:** Discuss these initial findings. Critique each other. "
            "Come to a consensus score. Err on the side of being critical (lower scores)."
        )
        

        # Moderator must output the final JSON summary
        moderator_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            response_format=ModeratorConsensus,
            key=OPENAI_KEY
        )

        moderator = AssistantAgent(
            name="Moderator", 
            model_client=moderator_client, 
            system_message="Listen to the debate. In your final turn, output the consensus JSON."
        )

        # 4. Define Team (RoundRobin)
        # Order: Sem -> Sty -> Cul -> Sem -> Sty -> Cul -> Moderator (End)
        # 3 agents * 2 rounds = 6 turns. + 1 Moderator turn = 7 total.
        termination = MaxMessageTermination(max_messages=7)

        party = [x.get_agent() for x in self.linguists]

        debate_team = RoundRobinGroupChat(
            participants=[*party, moderator],
            termination_condition=termination
        )

        # 5. Run the Team
        # We pass the constructed context as the 'task'
        result = await debate_team.run(task=initial_context)

        # 6. Extract Result
        # The last message is from the Moderator (Structured JSON)
        final_msg = result.messages[-1]

        print("--- Debate ---")
        for message in result.messages:
            print(message)

        print("--- Conclusion ---")
        print(final_msg)
        
        try:
            # Pydantic structured output is returned as a serialized string in .content
            data = json.loads(final_msg.content)
            return data
        except Exception as e:
            print(f"❌ Error parsing debate output for {chapter} {verse}: {e}")
            return None

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
            
            if consensus_data:
                final_results.append(consensus_data)

        return pd.DataFrame(final_results)
    
async def run(df):
    debate = Debate()
    results = await debate.process_interleaved_dataframe(df)
    print(results)

if __name__ == "__main__":
    df = pd.read_csv("debate_analysis_results.csv")
    asyncio(run(df))