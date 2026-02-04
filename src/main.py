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
from data.interpresure import Interpresure
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


TRANSLATION_NAME = "ulb"
LANGUAGE = "es-419"
book_number = 58
book = "PHM"
chapter = 1
topic = "social"

filename = f"../lang/{LANGUAGE}/{book_number}-{book}.usfm"

# Parse a file
verses = parse_usfm_file(filename)

# Or use the parser directly
parser = UsfmParser()

with open(filename, "r", encoding="utf-8") as f:
    translation_usfm = f.read()
verses = parser.parse(translation_usfm)

# --- 1. Load Ground Truth Data ---
# This data file was generated in the previous step
try:
    df = pd.read_csv("../interpresure/interpresure_phm.csv")
except FileNotFoundError:
    print("Error: interpresure_phm.csv not found. Please ensure the data prep step was executed.")
    exit()

TRANSLATED_SCRIPTURE_DICT = {}

for _, row in df.iterrows():
    c = int(row['chapter'])
    verse = int(row['verse'])
    ref = f"{book} {chapter}:{verse}"
    text = verses[ref]

    TRANSLATED_SCRIPTURE_DICT[(c, verse)] = text

linguist_models = ["gemini-3-pro-preview", "gpt-5.2"]

async def run_debate(initial_analysis, interpresure, topic):
    debate = Debate(linguist_models)
    return await debate.process_interleaved_dataframe(initial_analysis, interpresure, topic)

from report.coalesce import coalesce_csvs
async def run_analysis():
    os.makedirs(f"../out/{LANGUAGE}/{book}/", exist_ok=True)

    opening_statement_file = f"../out/{LANGUAGE}/{book}/{book}_opening_statements.csv"
    debate_file = f"../out/{LANGUAGE}/{book}/{book}_debate_output.csv"
    final_output_file = f"../out/{LANGUAGE}/{book}/{book}_final_output.json"

    interpresure = Interpresure(book, 1)

    #initial = await LinguisticAnalysis(interpresure, topic, linguist_models, "gpt-5-mini").run(TRANSLATED_SCRIPTURE_DICT, book, topic, 1)
    #initial = pd.read_csv(opening_statement_file)
    #debate = await run_debate(initial, interpresure, topic)


    # initial.to_csv(opening_statement_file)
    # debate.to_csv(debate_file)

    annotation_columns = interpresure.get_topic_columns(topic)
    coalesce_csvs(individual_path=opening_statement_file, debate_path=debate_file, output_path=final_output_file, interpresure=interpresure, topic=topic, book=book, translation_title=TRANSLATION_NAME, translation_language=LANGUAGE, translation_usfm=translation_usfm)

if __name__ == "__main__":
    # df = pd.read_csv("debate_analysis_results.csv")
    asyncio.run(run_analysis())

    # coalesce_csvs("debate_analysis_results.csv", "debate_results.csv", "out.json")
    # coalesce_csvs("debate_analysis_results.csv", "debate_output.csv", "out2.json")