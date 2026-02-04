import time
import asyncio
import pandas as pd
import json
from typing import List
from pydantic import BaseModel, Field

from agents.eli5 import Eli5Agent
from agents.linguist import LinguistAgent
from agents.secretary import SecretaryAgent
from data.interpresure import Interpresure
from model.config import get_config_for_model

# AutoGen 0.4+ Imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

class LinguistTurn(BaseModel):
    """The structured output for a linguist's turn in the debate."""
    agent_name: str = Field(description="The name of the linguist speaking.")
    argument: str = Field(description="Your Markdown formatted critique, in English. Reference specific peers if agreeing/disagreeing. If this is a closing statement, include all specific details as to justify your score, even if the idea originated from a peer. This field should be a string containing Markdown formatted text.")
    proposed_score: int = Field(description="The score (1-10) you currently advocate for.")

class ModeratorTurn(BaseModel):
    """The output from the moderator."""
    intervene: bool = Field(description="Whether the moderator is stepping in to intervene. False if there is no need to intervene.")
    violators: List[str] = Field(description="The names of the participants who require intervention. Empty if there is no need to intervene.")
    feedback: str = Field(description="Markdown formatted feedback to give the debate participant if there is an intervention. Empty if there is no need to intervene.")

class DebateCommentator(BaseModel):
    """The final consensus output from the moderator."""
    chapter: int = Field(description="The chapter debated.")
    verse: str = Field(description="The verse debated.")
    final_consensus_score: int = Field(description="The final agreed-upon score (integer). Choose the lowest score in the event of non-consensus.")
    consensus_summary: str = Field(description="A detailed explanation of why this score was chosen.")
    closing_statements: List[str] = Field(description="Closing remarks from the participants in markdown, beginning with a section heading of the participant name.")

class Debate:

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

    def __init__(self, linguist_models: list[str], secretary_model = "gpt-5.2", eli5_model = "gpt-5.2", moderator_model="gpt-5-mini"):
        self.moderator_model = moderator_model
        configs = [get_config_for_model(model) for model in linguist_models]
        self.linguists = [LinguistAgent(f"{model_config['name'].upper()}_LINGUIST", model_config["model"], model_config["key"], model_config["base_url"], self.task_description, self.response_format) for model_config in configs]

        secretary_model_config = get_config_for_model(secretary_model)
        eli5_model_config = get_config_for_model(eli5_model)

        self.secretary = SecretaryAgent("SECRETARY", secretary_model_config["model"], secretary_model_config["key"], secretary_model_config["base_url"], response_format=self.response_format)
        self.eli5agent = Eli5Agent("ELI5_SUMMARIZER", eli5_model_config["model"], eli5_model_config["key"], eli5_model_config["base_url"], response_format=self.response_format)

    async def _run_single_verse_debate(self, group_df: pd.DataFrame, interpresure: Interpresure, topic: str):
        """
        Runs a RoundRobin debate for a single group of 3 dataframe rows (one verse).
        """
        # Extract metadata from the first row of the group
        chapter = group_df.iloc[0]['chapter']
        verse = group_df.iloc[0]['verse']
        greek_text = group_df.iloc[0]['greek_text']
        translation = group_df.iloc[0]['translation']
        annotation_columns = interpresure.get_topic_columns(topic)

        print(f"\n--- 🗣️  Initiating Debate for {chapter} {verse} ---")

        # 1. Construct the Context from the DataFrame Rows
        # We map the specific agent names to their previous independent analysis
        initial_context = f"## Debate Context for {chapter} {verse}\n"
        initial_context += f"**Greek Text** {greek_text}\n\n"
        initial_context += f"** Translation ** {translation}\n\n"

        for col in annotation_columns:
            initial_context += f"## **{col}** {group_df.iloc[0][col]}\n\n"

        initial_context += "### Initial Independent Analyses:\n"

        opening_statements = []
        
        for _, row in group_df.iterrows():
            opening_statements.append(row['model_analysis'])
            initial_context += (
                f"- **{row['agent_name']}** (Initial Score: {row['score']}): {row['model_analysis']}\n"
            )
        
        initial_context += (
            "\n**TASK:** Debate these initial findings. Critique each other. "
            "Come to a consensus score. Err on the side of being critical (lower scores)."
        )

        print("Debate Initial Context:\n" + initial_context)
        
        moderator_model_config = get_config_for_model(self.moderator_model)
        # Moderator must output the final JSON summary
        moderator_client = OpenAIChatCompletionClient(
            model=moderator_model_config["model"],
            response_format=ModeratorTurn,
            key=moderator_model_config["key"]
        )

        pragmatic_annotations = "\n".join([f"- {x}: {group_df.iloc[0][x]} " for x in annotation_columns])

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
            {pragmatic_annotations}

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

        summary = await self._generate_summary(opening_statements, debate, closing_statements)
        eli5 = await self._simplified_summary(summary)

        row_data = {
            'chapter': chapter,
            'verse': verse,
            'greek_text': greek_text,
            'translation': translation,
            "debate": json.dumps(debate),
            "closing_statements": json.dumps(closing_statements),
            "summary": summary,
            "eli5": eli5
        } | { x: group_df.iloc[0][x] for x in annotation_columns}
        
        return pd.DataFrame([row_data])

    async def _generate_summary(self, opening_statements, debate, closing_statements):
        summary = await self.secretary.summarize(opening_statements, debate, closing_statements)
        return summary
    
    async def _simplified_summary(self, summary):
        summary = await self.eli5agent.eli5(summary)
        return summary
    
    async def process_interleaved_dataframe(self, df: pd.DataFrame, interpresure: Interpresure, topic: str):
        num_linguists = len(self.linguists)
        final_results = []

        # GROUP BY: This automatically handles the "3 rows at a time" requirement 
        # regardless of whether the rows are perfectly sorted or mixed up.
        grouped_data = df.groupby(['chapter', 'verse'])

        async def run(group_df):
            try:
                # Run the async debate for this specific group
                time.sleep(3)
                consensus_data = await self._run_single_verse_debate(group_df, interpresure, topic)
                print(consensus_data)
                final_results.append(consensus_data)
            except Exception as e:
                print("Error! rate limit, trying again...", e)
                time.sleep(61 * 5)
                print("INFO: retrying")
                await run(group_df)

        for (chapter, verse), group_df in grouped_data:
            if len(group_df) != num_linguists:
                print(f"⚠️ {chapter} {verse}: Expected 2 analyses, found {len(group_df)}, iterating...")
                verse_group = group_df.groupby(group_df.index // num_linguists)
                for _, df in verse_group:
                    await run(df)
                
            else:
                await run(group_df)
            

        print("---------- FINAL RESULTS -----------")
        print(final_results)

        return pd.concat(final_results)
    
async def run(df):
    debate = Debate()
    results = await debate.process_interleaved_dataframe(df)
    print(results)

if __name__ == "__main__":
    df = pd.read_csv("debate_analysis_results.csv")
    asyncio(Debate(["gemini-3-pro-preview", "gpt-5.2"]).run(df))