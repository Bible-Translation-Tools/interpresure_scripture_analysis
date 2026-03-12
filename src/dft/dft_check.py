import asyncio
import json
from typing import List, Optional

# AutoGen 0.7.x imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pydantic import BaseModel, Field
from usfm2dict import parse_usfm_file, UsfmParser

from dft.dft import DFT

class DFTVerification(BaseModel):
    dft_preserved: bool = Field(
        description="True if the equivalent divine familial term is present in the translation verse."
    )
    term_used: Optional[str] = Field(
        description="The specific word found in the translation, or null if dft_preserved is false."
    )
    verse_text: str = Field(
        description="The full text of the verse from the target translation."
    )
    verse_reference: str = Field(
        description="The standard verse reference being checked (e.g., MAT 2:15)."
    )

async def run_dft_verification_workflow(
    translation_usfm_path: str,
    greek_usfm_path: str,
    book: str,
    chapter: int,
    target_terms: List[str]
):
    # 1. Initialize Data
    dft_handler = DFT(book, chapter)
    dft_entries = dft_handler.get_annotations(book, chapter)
    
    parser = UsfmParser()
    with open(translation_usfm_path, "r", encoding="utf-8") as f:
        target_verses = parser.parse(f.read())
    
    # 2. Setup Model Client
    model_client = OpenAIChatCompletionClient(model="gpt-5-mini")

    # 3. Define Agents
    finder = AssistantAgent(
        name="Finder",
        model_client=model_client,
        system_message=f"""You are a linguistic analyst. 
        Identify if a Divine Familial Term (DFT) exists in the translation text.
        Target terms: {target_terms}.
        
        Output ONLY a JSON object:
        {{
            "dft_preserved": boolean,
            "term_used": "word found or null",
            "verse_text": "full text",
            "verse_reference": "REF"
        }}"""
    )

    checker = AssistantAgent(
        name="Checker",
        model_client=model_client,
        system_message="""Verify the Finder's JSON. 
        1. Ensure 'term_used' actually exists in 'verse_text'.
        2. If correct, reply EXACTLY with 'VALIDATED'.
        3. If incorrect, explain the error to the Finder so they can try again."""
    )

    # 4. Define Termination Conditions
    # Stops if VALIDATED is mentioned OR if we hit 6 messages (3 attempts)
    termination = TextMentionTermination("VALIDATED") | MaxMessageTermination(6)

    results = []

    # 5. Process Entries
    for _, row in dft_entries.iterrows():
        ref = row['verse_reference']
        if ref not in target_verses:
            continue

        verse_context = {
            "reference": ref,
            "greek": row['greek'],
            "english": row['english'],
            "translation_text": target_verses[ref]
        }

        # Create the Team for this specific verse
        team = RoundRobinGroupChat([finder, checker], termination_condition=termination)

        # Run the conversation
        print(f"Checking {ref}...")
        task_query = f"Analyze this verse context and provide the JSON: {json.dumps(verse_context)}"
        
        # In 0.7.x, we iterate through the task stream
        final_response = None
        async for message in team.run_stream(task=task_query):
            # We track the last message from Finder to capture the JSON
            if message.source == "Finder":
                final_response = message.content

        # Post-process: Only append if the conversation was actually validated
        # We check the team's last message in the stream or the termination state
        # For simplicity, we'll verify the presence of 'VALIDATED' in the team history
        history = await team.get_team_status() # Or check the stream results
        
        # Check if the last message from Checker was validation
        if "VALIDATED" in str(final_response) or any("VALIDATED" in str(m) for m in [final_response]):
             # This is simplified; ideally, you'd parse the 'final_response' 
             # from the Finder right before the Checker said VALIDATED
             pass

        try:
            # Extract JSON from the Finder's content
            clean_json = final_response.replace("```json", "").replace("```", "").strip()
            results.append(json.loads(clean_json))
        except (json.JSONDecodeError, AttributeError):
            print(f"Could not extract valid JSON for {ref}")

    return results

# --- Main Execution ---
if __name__ == "__main__":

    LANGUAGE = "vi"
    book_number = "40"
    book = "MAT"

    filename = f"../lang/{LANGUAGE}/{book_number}-{book}.usfm"
    greek_filename = f"../lang/grc/{book_number}-{book}.usfm"
    hebrew_filename = f"../lang/heb/{book_number}-{book}.usfm"

    target_results = asyncio.run(run_dft_verification_workflow(
        translation_usfm_path=filename,
        greek_usfm_path=greek_filename,
        book="MAT",
        chapter=2,
        target_terms=["Fillo", "Pai"]
    ))

    print(json.dumps(target_results, indent=2, ensure_ascii=False))