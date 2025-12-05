import pandas as pd
import json
import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination

from usfm2dict import parse_usfm_file, UsfmParser
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENAI_API_KEY")


# Parse a file
verses = parse_usfm_file("lang/en/58-PHM.usfm")

# Or use the parser directly
parser = UsfmParser()
book = "PHM"
with open("lang/en/58-PHM.usfm", "r", encoding="utf-8") as f:
    content = f.read()
verses = parser.parse(content)

# --- 1. Load Ground Truth Data ---
# This data file was generated in the previous step
try:
    df = pd.read_csv("philemon_face_ground_truth.csv")
except FileNotFoundError:
    print("Error: philemon_face_ground_truth.csv not found. Please ensure the data prep step was executed.")
    exit()

# --- 2. Define LLM Configurations ---
# NOTE: Replace 'gpt-5' with the actual model name when available (e.g., 'gpt-4o').
# NOTE: Replace 'gemini-2.5-flash' with the actual Gemini model name supported by your Autogen setup.
# You MUST configure your Autogen environment to use the API keys for these services.




openai_client = OpenAIChatCompletionClient(model="gpt-5", api_key=key, seed=42)

# Primary Scorer - Focus on Quantification
scorer_gpt = AssistantAgent(
    name="Primary_Scorer_GPT",
    system_message="""
            You are a descriptive linguist and cross-cultural pragmatics expert. 
            Your task is to assign a score (1-5, where 5 is best) to the translation 
            based on how well it preserves the specific 'Face' act (e.g., Mitigate Negative Face) from the Greek source. 
            The entire conversation and final output MUST be in English. 
            Base your score ONLY on concrete lexical, grammatical, or rhetorical fidelity to the pragmatic goal, NOT theological opinion.
            """,
    model_client=openai_client
)

# Justifier/Cross-Checker - Focus on Qualitative Rigor
justifier_gemini = AssistantAgent(
    name="Validator_GPT",
    system_message="""You are a peer reviewer and rhetorical critic specializing in comparative linguistics. 
            Review the Primary Scorer's output. Your main task is to verify that the justification is purely linguistic, 
            explicitly referencing the features of the target language (e.g., honorifics, verb mood) and NOT theological bias. 
            If the justification is solid, provide the final structured result. 
            If it contains bias, critique it and prompt the Scorer to revise. 
            Your final confirmed response MUST contain a line starting with 'FINAL_ANALYSIS:""",
    model_client=openai_client
)

termination = TextMentionTermination("FINAL_ANALYSIS:")

groupchat = RoundRobinGroupChat(
    [scorer_gpt, justifier_gemini], 
    max_turns=15,
    termination_condition=termination
)

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

async def run_analysis():
    for _, row in df.head(5).iterrows():  # using head(5) for quick demo; remove head() when full run
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
            "FIRST, assign a score (1-5) to the translation for its fidelity to the 'Face' goal. "
            "SECOND, justify your score by identifying the specific linguistic feature (e.g., verb tense, politeness markers) "
            "that either succeeds or fails. The PRIMARY_SCORER_GPT must output the score and initial justification."
        )

        print(f"\n--- Analyzing Philemon {chapter}:{verse} ---")
        task_result = await groupchat.run(task=prompt)

        # Grab the last message content
        final_msg = task_result.messages[-1].content

        print("-- Conversation --")

        for message in task_result.messages:
            print(message.content)

        if final_msg.startswith("FINAL_ANALYSIS:"):
            results.append({
                "Chapter": chapter,
                "Verse": verse,
                "Face_Annotation": row['Face'],
                "Final_Analysis": final_msg[len("FINAL_ANALYSIS:"):].strip()
            })
        else:
            results.append({
                "Chapter": chapter,
                "Verse": verse,
                "Face_Annotation": row['Face'],
                "Final_Analysis": "ERROR: Termination message not found or chat failed."
            })

    # --- 6. Final Reporting ---
    final_df = pd.DataFrame(results)
    final_df.to_csv("autogen_face_analysis_results.csv", index=False)
    print("\n--- AutoGen Script Finished ---")
    print("Results saved to autogen_face_analysis_results.csv")
    print(final_df.to_markdown(index=False, numalign="left", stralign="left"))

    # Close the model clients
    await openai_client.close()

if __name__ == "__main__":
    asyncio.run(run_analysis())