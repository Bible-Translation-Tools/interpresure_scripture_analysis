import json
from autogen_agentchat.teams import RoundRobinGroupChat
from pandas import DataFrame
from agents.analyst import PragmaticAnalystReview
from agents.critic import CriticAgent, CriticReview
from agents.linguist import LinguistAgent, LinguistReview
from data.interpresure import Interpresure
import os

from model.config import get_config_for_model

def parse_critic_output(json_str: str) -> CriticReview | None:
    """Parses the Critic's JSON output."""
    try:
        data = json.loads(json_str.strip())
        return CriticReview(**data)
    except Exception as e:
        print(f"Error parsing Critic JSON (Attempted to parse: '{json_str.strip()[:50]}...'): {e}")
        return None

class PragmaticAnalysis:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "pragmatic_review",
            "strict": True,
            "schema": PragmaticAnalystReview.model_json_schema()
        }
    }

    def __init__(
        self, 
        interpresure: Interpresure, 
        linguist_model: str, 
        critic_model: str,
        biblical_language
    ):
        
        self.biblical_language = biblical_language

        self.interpresure = interpresure

        self.task_description = f"""
            Evaluate the following Bible translation with respect to the pragmatic goal, focusing primarily on whether the implicit meaning—that is, what is communicated between the lines, in addition to what is stated directly—is preserved in the translation. 
            Provide a list of strengths, weaknesses, and suggestions for the translation based on the expert annotations provided.
            Your analysis MUST be in English. 
            Base your analysis ONLY on concrete lexical, grammatical, or rhetorical fidelity to the pragmatic goal, NOT theological opinion.
            """

        print("Independent Analysis Task Description:\n" + self.task_description)

        config = get_config_for_model(linguist_model)
        self.linguist = LinguistAgent(f"{config['name'].upper()}_LINGUIST", config["model"], config["key"], config["base_url"], self.task_description, self.response_format)
        self.critic = CriticAgent(critic_model, biblical_language)

        self.chat = RoundRobinGroupChat(
            participants=[self.linguist.agent, self.critic.agent],
            max_turns=1  # one agent turn per run()
        )


    async def _perform_analysis_and_review(self, analysis_prompt):

        print(f"\n--- Starting Independent Analysis for {self.linguist.agent.name} ---")

        # --- Step 1: Independent Analysis (linguist) ---
        # Pass the user prompt as 'task' so the linguist receives it and speaks.
        result = await self.chat.run(task=analysis_prompt)
        # result.messages is the list of published messages from the team run.
        critique_for_review = result.messages[-1].content
        # (Depending on agent implementations you may prefer to inspect by agent name.)

        # --- Review Loop ---
        review_round = 0
        max_review_rounds = 3

        while review_round < max_review_rounds:
            print(f"\n--- Review Round {review_round + 1} ---")

            # Ask critic to review (force JSON schema)
            critic_instruction = (
                f"{self.critic.agent.name}: Review the following analysis and respond ONLY in JSON.\n\n"
                f"{json.dumps(CriticReview.model_json_schema())}\n\n"
                f"Analysis:\n{critique_for_review}"
            )

            review_result = await self.chat.run(task=critic_instruction)
            critic_output_str = review_result.messages[-1].content
            critic_review = parse_critic_output(critic_output_str)

            if not critic_review:
                print("❌ Critic failed to produce valid JSON. Accepting current analysis.")
                return critique_for_review

            if critic_review.accepted:
                print(f"✅ Analysis Accepted: {critic_review.reasoning}")
                return critique_for_review

            # If rejected → tell linguist to revise
            print(f"🛑 Analysis Rejected: {critic_review.reasoning}")

            revision_prompt = (
                f"{self.linguist.agent.name}: Your previous analysis was rejected because:\n"
                f"'{critic_review.reasoning}'.\n"
                "Please revise your critique. "
            )

            revision_result = await self.chat.run(task=revision_prompt)
            critique_for_review = revision_result.messages[-1].content
            review_round += 1

        print("⚠️ Max review rounds reached — returning last version.")
        return critique_for_review
    
    def task_prompt(self, pragmatic_annotations, translation, original_verse, biblical_language="greek"):
        prompt = (
            f"# GROUND TRUTH ANALYSIS:\n"
            f"{biblical_language.capitalize()} Verse: \"{original_verse}\"\n"
            f"Expert Pragmatic Annotations: \"{pragmatic_annotations}\"\n"
            "# TASK:\n"
            f"Translation: \"{translation}\"\n\n"
            "Provide your analysis in English, with sectioned markdown lists of (1) strengths, (2) weaknesses, and (3) suggestions regarding the translation's performance against these pragmatic objectives, and a score from 1 to 10."
        )
        
        return prompt
    
    async def run(
        self, 
        translated_scripture_dict: dict,
        biblical_scripture_dict: dict, 
        book, 
        sample: int = None,
    ) -> DataFrame:
        topic = "general"

        print(f"Running analysis for: {book}, {topic}")

        results = []
        df = self.interpresure.get_annotations(topic)

        grouped_data = df.groupby(['chapter', 'verse'])

        processed = 0

        for (chapter, verse), group_df in grouped_data:
            if sample is not None and processed >= sample:
                break

            if (chapter, verse) not in translated_scripture_dict:
                print(f"ERROR! Chapter {chapter} Verse {verse} not in the translation corpus!")
                continue

            if (chapter, verse) not in biblical_scripture_dict:
                print(f"ERROR! Chapter {chapter} Verse {verse} not in the {self.biblical_language} corpus!")
                continue

            pragmatic_annotations = self.interpresure.get_annotations_markdown(topic, int(chapter), int(verse))
            translated_text = translated_scripture_dict[(chapter, verse)]
            biblical_text = biblical_scripture_dict[(chapter, verse)]
        

            linguist = self.linguist
            critique = await self._perform_analysis_and_review(
                self.task_prompt(pragmatic_annotations, translated_text, biblical_text, self.biblical_language)
            )

            print(f"--- {linguist.name} Analysis {book} {chapter}:{verse} ---")
            print(critique)

            critique = json.loads(critique)

            strengths = critique["strengths"]
            weaknesses = critique["weaknesses"]
            suggestions = critique["suggestions"]

            results.append(
                {
                    "model": linguist.model_name,
                    "agent_name": linguist.name,
                    "chapter": chapter,
                    "verse": verse,
                    "biblical_text": biblical_text,
                    "translation": translated_text,
                    # "notes": row['notes'],
                    "score": critique["score"],
                    "model_analysis": f"# Strengths:\n{strengths}\n\n# Weaknesses:\n{weaknesses}\n\n# Suggestions:\n{suggestions}"
                } 
            )

            processed += 1

        critique = await self._perform_analysis_and_review("Given the analysis of all verses within this chapter, create a final chapter overview.")
        critique = json.loads(critique)

        strengths = critique["strengths"]
        weaknesses = critique["weaknesses"]
        suggestions = critique["suggestions"]
        verses_to_review = critique["verses_to_review"]

        results.append(
            {
                "model": linguist.model_name,
                "agent_name": linguist.name,
                "chapter": chapter,
                "verse": 0,
                "biblical_text": biblical_text,
                "translation": translated_text,
                # "notes": row['notes'],
                "score": critique["score"],
                "model_analysis": f"# Strengths:\n{strengths}\n\n# Weaknesses:\n{weaknesses}\n\n# Suggestions:{suggestions}\n\n# Verses to Review:\n{verses_to_review}"
            } 
        )

        final_df = DataFrame(results)
        final_df.to_csv("pragmatic_analysis.csv", index=False)
        print("\n--- AutoGen Script Finished ---")
        print("Results saved to pragmatic_analysis.csv")
        print(final_df.to_markdown(index=False, numalign="left", stralign="left"))
        return final_df
