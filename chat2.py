import pandas as pd
import json
import os
import asyncio
from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import StopMessage

from usfm2dict import parse_usfm_file, UsfmParser

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
key = os.getenv("GEMINI_API_KEY")

#key = os.getenv("OPENAI_API_KEY")


LANGUAGE = "en"
book_number = 58
book = "PHM"
filename = f"lang/{LANGUAGE}/{book_number}-{book}.usfm"

# Parse a file
verses = parse_usfm_file(filename)

# Or use the parser directly
parser = UsfmParser()

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()
verses = parser.parse(content)

# --- 1. Load Ground Truth Data ---
# This data file was generated in the previous step
try:
    df = pd.read_csv("philemon_face_ground_truth.csv")
except FileNotFoundError:
    print("Error: philemon_face_ground_truth.csv not found. Please ensure the data prep step was executed.")
    exit()


class CriticDecision(BaseModel):
    """Structured schema for critic's decision."""
    approved_by_critic: bool = Field(..., description="Whether the result is approved by the critic.")
    score: int = Field(..., description="The translation's score from the PRIMARY_LINGUIST (1-10, where 10 is best).")
    score_justification: str = Field(..., description="The reason for the translation's score from the PRIMARY_LINGUIST")
    critique: str = Field(..., description="Reason for the rejection by the critic and feedback for the PRIMARY_LINGUIST for correction.")


CLIENT_MODEL = "gemini-3-pro-preview" # "gpt-5"

conversation_client = OpenAIChatCompletionClient(
    api_type="openai",
    model=CLIENT_MODEL,
    base_url=GOOGLE_BASE_URL,
    model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
    api_key=key, 
    #seed=42
)

critic_client = OpenAIChatCompletionClient(
    api_type="openai",
    model=CLIENT_MODEL, 
    base_url=GOOGLE_BASE_URL, 
    model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
    api_key=key, 
    #seed=42, 
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "critic_decision",
            "schema": CriticDecision.model_json_schema()
        }
    }
)

class CriticApproval(TerminationCondition):
    """
    Terminates when:
      - Critic outputs JSON with `"approved": true"`, OR
      - Each agent reaches `max_turns_per_agent`
    """

    def __init__(self, critic_name: str):
        super().__init__()
        self.critic_name = critic_name
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages):        
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        for msg in messages:
            sender = getattr(msg, "source", None)
            content = getattr(msg, "content", None)
            if not sender or not content:
                continue

            # Check if critic approved
            if sender == self.critic_name:
                try:
                    data = json.loads(content)
                    decision = CriticDecision.model_validate(data)
                    if decision.approved_by_critic:
                        self._terminated = True
                        print("Approved!")
                        return StopMessage(
                            content=f"Critic approved response.", 
                            source="CriticApproval"
                        )
                except Exception:
                    # Ignore parse errors — critic may output partial text
                    print("Parse error?")
                    pass

        return None

    async def reset(self) -> None:
        """Reset state for reuse."""
        self._terminated = False

# --- 5. Looping and Analysis ---

NON_ENGLISH_TRANSLATION_DICT = {}

for _, row in df.iterrows():
    chapter = int(row['Chapter'])
    verse = int(row['Verse'])
    ref = f"{book} {chapter}:{verse}"
    text = verses[ref]

    NON_ENGLISH_TRANSLATION_DICT[(chapter, verse)] = text

results = []

print("--- Starting AutoGen Ensemble Analysis ---")


# Primary Scorer - Focus on Quantification
scorer_gpt = AssistantAgent(
    name="PRIMARY_LINGUIST",
    system_message="""
            You are a descriptive linguist and cross-cultural pragmatics expert. 
            Your task is to assign a score (1-10, where 10 is best) to the translation 
            based on how well it preserves the specific 'Face' act (e.g., Mitigate Negative Face) from the Greek source. 
            Err on the side of being more critical in the score.
            The entire conversation and final output MUST be in English. 
            Base your score ONLY on concrete lexical, grammatical, or rhetorical fidelity to the pragmatic goal, NOT theological opinion.
            """,
    model_client=conversation_client
)

# Justifier/Cross-Checker - Focus on Qualitative Rigor
critic_gpt = AssistantAgent(
    name="Validator_GPT",
    system_message="""You are a peer reviewer and rhetorical critic specializing in comparative linguistics. 
            Review the PRIMARY_LINGUIST's output. Your main task is to verify that the justification is purely linguistic, 
            explicitly referencing the features of the target language (e.g., honorifics, verb mood) and NOT theological bias.  
            If it contains bias, critique it and prompt the PRIMARY_LINGUIST to revise by setting approved_by_critic to false and providing the critique in the critique field. 
            If the justification is linguistically sound and does not contain bias, review if the score matches the justification.
            If the score does not match the justification (such as a score of 10 but a justification suggesting imperfections in the translation), set approved_by_critic to false and provide the feedback to the PRIMARY_LINGUIST in the critique field.
            If everything is good, set approved_by_critic to true, and provide the PRIMARY_LINGUIST's justification in the score_justification field.
            Respond strictly in valid JSON conforming to the given schema.
            """,
    model_client=critic_client
)

commentator_gpt = AssistantAgent(
    name="Commentator_GPT",
    system_message="""You are an expert in linguistics and talented at teaching complex subjects in simple terms. 
                You will review a linguistic analysis of a translation, and express that analysis such that a middle or high school student would be able to understand.
                Make sure that your output is in English, formatted in Markdown.
            """,
    model_client=conversation_client
)

async def get_agent_commentary(row, commentator_gpt) -> str:
    commentary = row["Final_Analysis"]
    question = f"Explain the following linguistic commentary to an average person: {commentary}"
    
    task_result = await commentator_gpt.run(task=question)
    
    if task_result and task_result.messages:
        return task_result.messages[-1].content
    else:
        return ""

import time

def process_commentary_column_sync(df, commentator_gpt):
    """
    Runs the agent tasks synchronously with a 30-second sleep between calls
    and updates the DataFrame.
    
    NOTE: This is a synchronous (blocking) function.
    """
    
    # --- 1. Prepare to store results ---
    results = []
    
    print(f"Starting {len(df)} synchronous agent calls with a 30-second delay between each...")
    
    # --- 2. Iterate through each row and call the synchronous function ---
    for index, row in df.iterrows():
        # Call the synchronous version of the function (assuming get_agent_commentary
        # is now a synchronous function or can be called synchronously)
        commentary = get_agent_commentary(row, commentator_gpt)
        results.append(commentary)
        
        # Introduce a 30-second sleep *after* processing the row, unless it's the last one
        if index < len(df) - 1:
            print(f"Completed commentary for index {index}. Pausing for 30 seconds...")
            time.sleep(30)
        else:
            print(f"Completed commentary for index {index}. No pause needed (last call).")
    
    # --- 3. Assign the results back to a new column ---
    # The 'results' list is in the same order as the DataFrame rows
    df["Commentary"] = results
    
    return df

async def run_analysis():
    for _, row in df.iterrows():  # using head(5) for quick demo; remove head() when full run

        roundtable = [scorer_gpt, critic_gpt]

        TURNS = 3
        MAX_TURNS = (len(roundtable) * TURNS) + 1 # Adds one for the instruction

        termination = CriticApproval("Validator_GPT") | MaxMessageTermination(max_messages=MAX_TURNS)

        groupchat = RoundRobinGroupChat(
            roundtable, 
            termination_condition=termination
        )

        chapter = row['Chapter']
        verse = row['Verse']

        if (chapter, verse) not in NON_ENGLISH_TRANSLATION_DICT:
            continue

        non_eng_text = NON_ENGLISH_TRANSLATION_DICT[(chapter, verse)]
        prompt = (
            f"ROLE: You are performing a cross-lingual pragmatic analysis.\n"
            "---------------------------------------------------------------------------------\n"
            f"GROUND TRUTH ANALYSIS:\n"
            f"1. Context: Philemon {chapter}:{verse}\n"
            f"2. Greek Text: \"{row['GreekText']}\"\n"
            f"3. Pragmatic Goal ('Face'): \"{row['Face']}\"\n"
            f"4. Expert Rationale (The Constraint): \"{row['Notes']}\"\n\n"
            "TASK:\n"
            f"Evaluate the following translation against the Greek pragmatic goal and rationale.\n"
            f"Translation: \"{non_eng_text}\"\n\n"
            "FIRST, assign a score (1-10) to the translation for its fidelity to the 'Face' goal. "
            "SECOND, justify your score by identifying the specific linguistic feature (e.g., verb tense, politeness markers) "
            "that either succeeds or fails. The PRIMARY_SCORER_GPT must output the score and initial justification."
        )

        print(f"\n# Analyzing Philemon {chapter}:{verse}")
        task_result = await groupchat.run(task=prompt)

        # Grab the last message content
        final_msg = task_result.messages[-1].content

        print("\n")

        for message in task_result.messages:
            print(f"\n### {message.source}\n")
            print(message.content)

        print("-----------")

        # Parse final output
        critic_decision = CriticDecision.model_validate_json(final_msg)
        

        if critic_decision.approved_by_critic:
            results.append({
                "Chapter": chapter,
                "Verse": verse,
                "Face_Annotation": row['Face'],
                "Score": critic_decision.score,
                "Final_Analysis": critic_decision.score_justification
            })
        else:
            results.append({
                "Chapter": chapter,
                "Verse": verse,
                "Face_Annotation": row['Face'],
                "Score": 1,
                "Final_Analysis": "ERROR: Termination message not found or chat failed."
            })

    # --- 6. Final Reporting ---
    final_df = pd.DataFrame(results)
    final_df.to_csv("autogen_face_analysis_results.csv", index=False)
    print("\n--- AutoGen Script Finished ---")
    print("Results saved to autogen_face_analysis_results.csv")
    print(final_df.to_markdown(index=False, numalign="left", stralign="left"))

    final_df = await process_commentary_column_sync(final_df, commentator_gpt)
    final_df.to_csv("vi_autogen_face_analysis_results_with_commentary.csv", index=False)


    # Close the model clients
    await conversation_client.close()

async def run_commentary(): 
    final_df = pd.read_csv("vi_autogen_face_analysis_results2.csv")
    final_df = await process_commentary_column_sync(final_df, commentator_gpt)
    final_df.to_csv("vi_autogen_face_analysis_results_with_commentary2.csv", index=False)
    print("DONE!")
    

if __name__ == "__main__":
    #asyncio.run(run_commentary())
    
    asyncio.run(run_analysis())