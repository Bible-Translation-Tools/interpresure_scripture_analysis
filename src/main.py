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
from teams.debate import Debate
from teams.analysis import LinguisticAnalysis
from agents.critic import CriticAgent
from agents.linguist import LinguistAgent, LinguistReview

from usfm2dict import parse_usfm_file, UsfmParser

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


LANGUAGE = "en"
book_number = 58
book = "PHM"
filename = f"../lang/{LANGUAGE}/{book_number}-{book}.usfm"

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
    df = pd.read_csv("../philemon_face_ground_truth.csv")
except FileNotFoundError:
    print("Error: philemon_face_ground_truth.csv not found. Please ensure the data prep step was executed.")
    exit()

NON_ENGLISH_TRANSLATION_DICT = {}

for _, row in df.iterrows():
    chapter = int(row['Chapter'])
    verse = int(row['Verse'])
    ref = f"{book} {chapter}:{verse}"
    text = verses[ref]

    NON_ENGLISH_TRANSLATION_DICT[(chapter, verse)] = text

results = []

async def run_initial_analysis():
    task_description = """
    Your task is to assign a score (1-10, where 10 is best) to the translation 
    based on how well it preserves the specific 'Face' act (e.g., Mitigate Negative Face) from the Greek source. 
    Err on the side of being more critical in the score.
    The entire conversation and final output MUST be in English. 
    Base your score ONLY on concrete lexical, grammatical, or rhetorical fidelity to the pragmatic goal, NOT theological opinion.
    """

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "linguist_review",
            "schema": LinguistReview.model_json_schema()
        }
    }
    
    linguists = [
        LinguistAgent("GEMINI_LINGUIST", "gemini-3-pro-preview", GEMINI_KEY, GOOGLE_BASE_URL, task_description, response_format),
        LinguistAgent("GPT5_LINGUIST", "gpt-5.1", OPENAI_KEY, None, task_description, response_format)
    ]

    # linguists = [
    #     LinguistAgent("GEMINI_LINGUIST", "gemini-2.0-flash-lite", GEMINI_KEY, GOOGLE_BASE_URL, task_description, response_format),
    #     LinguistAgent("GPT5_LINGUIST", "gpt-4o", OPENAI_KEY, None, task_description, response_format)
    # ]

    critic = CriticAgent()

    for _, row in df.iterrows():  # using head(5) for quick demo; remove head() when full run
    
        chapter = row['Chapter']
        verse = row['Verse']

        if (chapter, verse) not in NON_ENGLISH_TRANSLATION_DICT:
            continue

        translated_text = NON_ENGLISH_TRANSLATION_DICT[(chapter, verse)]
        for linguist in linguists:
            analysis = LinguisticAnalysis(
                linguist.get_agent(), 
                critic.get_agent()
            )
            critique = await analysis.perform_analysis_and_review(
                linguist._construct_prompt(chapter, verse, row['GreekText'], row['Face'], row['Notes'], translated_text)
            )

            print(f"--- {linguist.name} Analysis {book} {chapter}:{verse} ---")
            print(critique)

            critique = json.loads(critique)
            results.append({
                "Model": linguist.model_name,
                "Agent_Name": linguist.name,
                "Chapter": chapter,
                "Verse": verse,
                "Greek_Text": row['GreekText'],
                "Translation": translated_text,
                "Face_Annotation": row['Face'],
                "Notes": row['Notes'],
                "Score": critique["score"],
                "Model_Analysis": critique["reasoning"]
            })

    final_df = pd.DataFrame(results)
    final_df.to_csv("opening_statements.csv", index=False)
    print("\n--- AutoGen Script Finished ---")
    print("Results saved to autogen_face_analysis_results.csv")
    print(final_df.to_markdown(index=False, numalign="left", stralign="left"))
    return final_df


async def run_debate(initial_analysis):
    debate = Debate()
    return await debate.process_interleaved_dataframe(initial_analysis)

from report.coalesce import coalesce_csvs
async def run_analysis():
    initial = await run_initial_analysis()
    debate = await run_debate(initial)

    os.makedirs("../out/", exist_ok=True)

    opening_statement_file = "../out/opening_statements.csv"
    debate_file = "../out/debate_output.csv"
    final_output_file = "../out/final_output.json"

    initial.to_csv(opening_statement_file)
    debate.to_csv(debate_file)

    coalesce_csvs(opening_statement_file, debate_file, final_output_file)

if __name__ == "__main__":
    # df = pd.read_csv("debate_analysis_results.csv")
    asyncio.run(run_analysis())

    # coalesce_csvs("debate_analysis_results.csv", "debate_results.csv", "out.json")
    # coalesce_csvs("debate_analysis_results.csv", "debate_output.csv", "out2.json")