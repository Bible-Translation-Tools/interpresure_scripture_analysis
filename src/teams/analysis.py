import json
from autogen_agentchat.teams import RoundRobinGroupChat
from pandas import DataFrame
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

class LinguisticAnalysis:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "linguist_review",
            "schema": LinguistReview.model_json_schema()
        }
    }

    def __init__(self, interpresure: Interpresure, topic: str, linguist_models: list[str], critic_model: str):

        self.interpresure = interpresure

        topic_title = interpresure.get_topic_title(topic)
        topic_goal = interpresure.get_topic_goal(topic)
        topic_description = interpresure.get_topic_description(topic)
        categories = interpresure.get_topic_categories(topic)

        self.task_description = f"""
            Your task is to assign a score (1-10, where 10 is best) to the translation with respect to {topic_title} from the Greek source.
            {topic_description}.
            {topic_title} includes the following categories: {categories}.
            {topic_goal}.
            Err on the side of being more critical in the score.
            The entire conversation and final output MUST be in English. 
            Base your score ONLY on concrete lexical, grammatical, or rhetorical fidelity to the pragmatic goal, NOT theological opinion.
            """

        print("Independent Analysis Task Description:\n" + self.task_description)

        configs = [get_config_for_model(model) for model in linguist_models]
        self.linguists = [LinguistAgent(f"{model_config['name'].upper()}_LINGUIST", model_config["model"], model_config["key"], model_config["base_url"], self.task_description, self.response_format) for model_config in configs]
        self.critic = CriticAgent(critic_model)


    async def _perform_analysis_and_review(self, linguist, analysis_prompt):
        linguist_agent = linguist.agent
        critic_agent = self.critic.agent

        chat = RoundRobinGroupChat(
            participants=[linguist_agent, critic_agent],
            max_turns=1  # one agent turn per run()
        )

        print(f"\n--- Starting Independent Analysis for {linguist_agent.name} ---")

        # --- Step 1: Independent Analysis (linguist) ---
        # Pass the user prompt as 'task' so the linguist receives it and speaks.
        result = await chat.run(task=analysis_prompt)
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
                f"{critic_agent.name}: Review the following analysis and respond ONLY in JSON.\n\n"
                f"{json.dumps(CriticReview.model_json_schema())}\n\n"
                f"Analysis:\n{critique_for_review}"
            )

            review_result = await chat.run(task=critic_instruction)
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
                f"{linguist_agent.name}: Your previous analysis was rejected because:\n"
                f"'{critic_review.reasoning}'.\n"
                "Please revise your critique to ensure it is based solely on verifiable linguistic principles. "
                "Retain the 'Score: [N]' format."
            )

            revision_result = await chat.run(task=revision_prompt)
            critique_for_review = revision_result.messages[-1].content
            review_round += 1

        print("⚠️ Max review rounds reached — returning last version.")
        return critique_for_review
    
    async def run(self, translated_scripture_dict: dict, book, topic, sample: int = None) -> DataFrame:
        print(f"Running analysis for: {book}, {topic}")

        results = []
        df = self.interpresure.get_annotations(topic)

        if sample is not None: 
            iterable = df.head(sample).iterrows() 
        else: 
            iterable = df.iterrows()

        for _, row in iterable:
        
            chapter = row['chapter']
            verse = row['verse']

            if (chapter, verse) not in translated_scripture_dict:
                continue

            translated_text = translated_scripture_dict[(chapter, verse)]

            pragmatic_annotations = "\n".join([f"- {x}: {row[x]} " for x in self.interpresure.get_topic_columns(topic)])

            for linguist in self.linguists:
                critique = await self._perform_analysis_and_review(
                    linguist,
                    linguist._construct_prompt(chapter, verse, row['greek_text'], pragmatic_annotations, row['notes'], translated_text)
                )

                print(f"--- {linguist.name} Analysis {book} {chapter}:{verse} ---")
                print(critique)

                critique = json.loads(critique)

                results.append(
                    {
                        "model": linguist.model_name,
                        "agent_name": linguist.name,
                        "chapter": chapter,
                        "verse": verse,
                        "greek_text": row['greek_text'],
                        "translation": translated_text,
                        "notes": row['notes'],
                        "score": critique["score"],
                        "model_analysis": critique["reasoning"]
                    } |
                    { x: row[x] for x in self.interpresure.get_topic_columns(topic)}
                )

        final_df = DataFrame(results)
        final_df.to_csv("opening_statements.csv", index=False)
        print("\n--- AutoGen Script Finished ---")
        print("Results saved to autogen_face_analysis_results.csv")
        print(final_df.to_markdown(index=False, numalign="left", stralign="left"))
        return final_df
