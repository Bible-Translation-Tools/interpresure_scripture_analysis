import pandas as pd
import json
import os
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.base import TerminationCondition, TerminatedException
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import StopMessage
from data.interpresure import Interpresure
from report import convert_pragmatic_analysis
from teams.debate import Debate
from teams.analysis import LinguisticAnalysis
from agents.critic import CriticAgent
from agents.linguist import LinguistAgent, LinguistReview

from usfm2dict import parse_usfm_file, UsfmParser

from dotenv import load_dotenv
import os

from teams.pragmatic_analysis import PragmaticAnalysis
from teams.summarize import SummarizeDebate
from report.compare_pragmatic_analysis import compare_pragmatic_analysis_files, print_comparison_summary

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


TRANSLATION_NAME = "ulb"
LANGUAGE = "en"
book_number = 19
book = "PSA"
chapter = 145

biblical_language = "hebrew"
ANALYSIS_MODE = "zero-shot"

filename = f"../lang/{LANGUAGE}/{book_number}-{book}.usfm"
greek_filename = f"../lang/grc/{book_number}-{book}.usfm"
hebrew_filename = f"../lang/heb/{book_number}-{book}.usfm"

# Parse a file
verses = parse_usfm_file(filename)

# Or use the parser directly
parser = UsfmParser()

with open(filename, "r", encoding="utf-8") as f:
    translation_usfm = f.read()

if biblical_language == "greek":
    with open(greek_filename, "r", encoding="utf-8") as f:
        greek_usfm = f.read()
    greek_verses = parser.parse(greek_usfm)
    GREEK_SCRIPTURE_DICT = {}
else:
    with open(hebrew_filename, "r", encoding="utf-8") as f:
        hebrew_usfm = f.read()
    hebrew_verses = parser.parse(hebrew_usfm)
    HEBREW_SCRIPTURE_DICT = {}

verses = parser.parse(translation_usfm)


interpresure = Interpresure(book, chapter)
df = interpresure.get_annotations("general")

TRANSLATED_SCRIPTURE_DICT = {}

for _, row in df.iterrows():
    c = int(row['chapter'])
    verse = int(row['verse'])
    ref = f"{book} {chapter}:{verse}"
    text = verses[ref]

    if biblical_language == "greek":
        greek_text = greek_verses[ref]
        GREEK_SCRIPTURE_DICT[(c, verse)] =  greek_text
    else:
        hebrew_text = hebrew_verses[ref]
        HEBREW_SCRIPTURE_DICT[(c, verse)] =  hebrew_text

    TRANSLATED_SCRIPTURE_DICT[(c, verse)] = text


linguist_model = "gpt-5.2"


async def run_analysis(
    analysis_mode: str = ANALYSIS_MODE,
):
    output_dir = Path(f"../out/{LANGUAGE}/{book}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_slug = analysis_mode.replace("-", "_")
    final_output_file = output_dir / f"{LANGUAGE}_{book}_pragmatics_analysis_{mode_slug}.json"

    opening_statement_file = output_dir / f"{book}_{mode_slug}_pragmatic_analysis.csv"

    interpresure = Interpresure(book, chapter)
    verse_records = []
    use_expert_materials = analysis_mode != "zero-shot"

    for _, row in df.iterrows():
        c = int(row["chapter"])
        verse = int(row["verse"])
        ref = f"{book} {chapter}:{verse}"
        verse_record = {
            "book": book.upper(),
            "chapter": c,
            "verse": verse,
            "translation_text": verses[ref],
            "biblical_text": greek_verses[ref] if biblical_language == "greek" else hebrew_verses[ref],
        }
        if use_expert_materials:
            verse_record["pragmatic_annotations"] = interpresure.get_annotations_markdown("general", c, verse)
        verse_records.append(verse_record)

    await PragmaticAnalysis(
        linguist_model,
        "gpt-5-mini",
        biblical_language,
        analysis_mode=analysis_mode,
        use_expert_materials=use_expert_materials,
    ).run(
        verse_records,
        output_csv_path=opening_statement_file,
    )

    eval_file = output_dir / f"{book}_{mode_slug}_general_analysis.json"
    eval = convert_pragmatic_analysis.convert_pragmatic(
        individual_path=opening_statement_file,
        output_path=eval_file,
        interpresure=interpresure,
        book=book,
    )
    final = finalize(final_output_file, LANGUAGE, TRANSLATION_NAME, translation_usfm, [eval])

    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, cls=NumpyBoolEncoder)

    return final_output_file


def compare_analysis_outputs(left_path: Path, right_path: Path, output_path: Path | None = None):
    report = compare_pragmatic_analysis_files(Path(left_path), Path(right_path))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyBoolEncoder)

    print_comparison_summary(report)
    return report

def finalize(outpath, translation_language, translation_title, translation_usfm, evaulations):
    final = {
        "translation": {
            "title": translation_title,
            "language": translation_language,
            "usfm": translation_usfm
        },
        "evaluation": evaulations
    }
    with open(outpath, "w", encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False, cls=NumpyBoolEncoder)

    return final

import numpy as np
class NumpyBoolEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NumPy booleans/numbers
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle standard booleans if they were somehow masked
        if isinstance(obj, bool):
            return str(obj) # Or just return obj to get JSON true/false
        return super().default(obj)


if __name__ == "__main__":
    # df = pd.read_csv("debate_analysis_results.csv")
    asyncio.run(run_analysis())

    # coalesce_csvs("debate_analysis_results.csv", "debate_results.csv", "out.json")
    # coalesce_csvs("debate_analysis_results.csv", "debate_output.csv", "out2.json")
