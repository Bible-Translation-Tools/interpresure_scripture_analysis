import json
from pathlib import Path
from typing import Any

from autogen_agentchat.teams import RoundRobinGroupChat
from pandas import DataFrame
from agents.analyst import PragmaticAnalystReview
from agents.critic import CriticAgent, CriticReview
from agents.linguist import LinguistAgent, LinguistReview

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
        linguist_model: str, 
        critic_model: str,
        biblical_language,
        *,
        analysis_mode: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        use_expert_materials: bool = False,
    ):
        
        self.biblical_language = biblical_language

        self.use_expert_materials = use_expert_materials
        self.analysis_mode = analysis_mode or ("few-shot" if self.use_expert_materials else "zero-shot")

        if self.use_expert_materials:
            self.task_description = """
                You are an expert Biblical Linguist and Translation Consultant. Your task is to evaluate a verse-by-verse translation from Biblical Greek/Hebrew into a Gateway Language. 

                **Response Language:** You MUST respond in English, regardless of the language of the source or target texts.

                **Inputs Provided:**
                1. The Original Text (Greek/Hebrew)
                2. The Target Translation (Gateway Language)
                3. Expert Ground Truth Annotations
                4. Discourse Analysis (BART displays if the Original Text is Greek)
                5. Syntax Trees (MACULA if the Original Text is Greek)

                **Evaluation Criteria:**
                Evaluate the translation based on pragmatics. Focus on implicatures, entailments, presuppositions, speech acts, and information structure (focus/emphasis). Determine if the intended meaning and unspoken assumptions of the original text are accurately communicated.
                In other words, the translation should communicate BOTH what is communicated in the line of the original text, but ALSO what is being communicated between the lines.
                Base your analysis ONLY on concrete lexical, grammatical, or rhetorical fidelity to pragmatics, NOT theological opinion.

                **Strict Resource Rule:** You MUST utilize the provided Ground Truth, BART, and MACULA data. If any significant pragmatic feature, emphasis, or implicature mentioned in the Ground Truth is NOT present or correctly handled in the Target Translation, the score MUST be lowered. 

                **Citation and Formatting Rules:**
                1. Use Markdown for the "Reasoning" and "Feedback" sections.
                2. In the "Reasoning" section, you may cite Greek/Hebrew words and phrases from the original text or target translation to support your technical analysis.
                3. In the "Feedback" section, you are strictly forbidden from using any Greek or Hebrew script, transliteration, or vocabulary. You MAY cite specific words or phrases from the Target Translation (Gateway Language) to make your suggestions clear.
                4. The Feedback must be in simple, layperson-friendly English without complex linguistic terminology.

                **Required Output Format:**
                * **Score:** [1 to 10]
                * **Reasoning:** [Markdown formatted. Explicitly explain how the Ground Truth, BART, and MACULA data influenced the score. Cite original and target text as needed.]
                * **Confidence:** [0 to 100]
                * **Feedback:** [Markdown formatted.]
                    * **Strengths:** [Simple terms]
                    * **Weaknesses:** [Simple terms]
                    * **Suggestions:** [Actionable, plain-language advice. Cite the Target Translation where helpful.]
                """

        else:
            self.task_description = """
            You are an expert Biblical Linguist and Translation Consultant. Your task is to evaluate a verse-by-verse translation from Biblical Greek/Hebrew into a Gateway Language. 

            **Response Language:** You MUST respond in English, regardless of the language of the source or target texts.

            **Inputs Provided:**
            1. The Original Text (Greek/Hebrew)
            2. The Target Translation (Gateway Language)

            **Evaluation Criteria:**
            Evaluate the translation based on your expert knowledge of Biblical language pragmatics (implicatures, entailments, presuppositions, speech acts, and information structure). Determine if the intended meaning and unspoken assumptions of the original text are accurately communicated.
            In other words, the translation should communicate BOTH what is communicated in the line of the original text, but ALSO what is being communicated between the lines.
            Base your analysis ONLY on concrete lexical, grammatical, or rhetorical fidelity to pragmatics, NOT theological opinion.

            **Citation and Formatting Rules:**
            1. Use Markdown for the "Reasoning" and "Feedback" sections.
            2. In the "Reasoning" section, you may cite Greek/Hebrew words and phrases from the original text or target translation to support your analysis.
            3. In the "Feedback" section, you are strictly forbidden from using any Greek or Hebrew script, transliteration, or vocabulary. You MAY cite specific words or phrases from the Target Translation (Gateway Language) to make your suggestions clear.
            4. The Feedback must be in simple, layperson-friendly English without complex linguistic terminology.

            **Required Output Format:**
            * **Score:** [1 to 10]
            * **Reasoning:** [Markdown formatted. Explain your pragmatic analysis of the original text and how the target translation handles those dynamics.]
            * **Confidence:** [0 to 100]
            * **Feedback:** [Markdown formatted.]
                * **Strengths:** [Simple terms]
                * **Weaknesses:** [Simple terms]
                * **Suggestions:** [Actionable, plain-language advice. Cite the Target Translation where helpful.]
            """

        print("Independent Analysis Task Description:\n" + self.task_description)

        config = get_config_for_model(linguist_model)
        linguist_api_key = api_key if api_key is not None else config["key"]
        linguist_base_url = base_url if base_url is not None else config["base_url"]
        self.linguist = LinguistAgent(
            f"{config['name'].upper()}_LINGUIST",
            config["model"],
            linguist_api_key,
            linguist_base_url,
            self.task_description,
            self.response_format,
        )
        self.critic = CriticAgent(critic_model, biblical_language, api_key=api_key, base_url=base_url)

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
    
    def task_prompt(
        self,
        pragmatic_annotations,
        translation,
        original_verse,
        biblical_language="greek",
        *,
        macula_tokens=None,
        bart_annotations=None,
    ):
        prompt_parts = [
            f"# GROUND TRUTH ANALYSIS:",
            f"{biblical_language.capitalize()} Verse: \"{original_verse}\"",
            f"Translation: \"{translation}\"",
        ]

        if pragmatic_annotations:
            prompt_parts.extend(
                [
                    "",
                    "Expert Pragmatic Annotations:",
                pragmatic_annotations,
                ]
            )

        if self.use_expert_materials and macula_tokens:
            prompt_parts.extend(
                [
                    "",
                    "Macula Tokens:",
                    json.dumps(macula_tokens, ensure_ascii=False, indent=2, default=str),
                ]
            )

        if self.use_expert_materials and bart_annotations:
            prompt_parts.extend(
                [
                    "",
                    "BART Annotations:",
                    json.dumps(bart_annotations, ensure_ascii=False, indent=2, default=str),
                ]
            )

        prompt_parts.extend(
            [
                "",
                "# TASK:",
                "Provide your analysis in English, with sectioned markdown lists of (1) strengths, (2) weaknesses, and (3) suggestions regarding the translation's performance against these pragmatic objectives, and a score from 1 to 10.",
            ]
        )

        return "\n".join(prompt_parts)

    def chapter_overview_prompt(self) -> str:
        return (
            "Given the analysis of all verses within this chapter, create a final chapter overview.\n"
            "Return the same JSON schema you used for the verse-level analyses.\n"
            "Use the score to summarize the chapter as a whole and list the verses that need the most review."
        )
    
    async def run(
        self, 
        verse_records: list[dict[str, Any]],
        sample: int = None,
        output_csv_path: str | Path | None = None,
    ) -> DataFrame:
        print(f"Running {self.analysis_mode} analysis")

        results = []

        processed = 0

        for verse_record in verse_records:
            if sample is not None and processed >= sample:
                break

            book = str(verse_record.get("book", "")).upper()
            chapter = int(verse_record["chapter"])
            verse = int(verse_record["verse"])
            translated_text = verse_record.get("translation_text", "")
            biblical_text = verse_record.get("biblical_text", "")
            pragmatic_annotations = verse_record.get("pragmatic_annotations", "")
        
            linguist = self.linguist

            task_prompt = self.task_prompt(
                    pragmatic_annotations,
                    translated_text,
                    biblical_text,
                    self.biblical_language,
                    macula_tokens=verse_record.get("macula_tokens"),
                    bart_annotations=verse_record.get("bart_annotations"),
                )

            print(f"--- Analysis Prompt {task_prompt} ---")

            critique = await self._perform_analysis_and_review(
                task_prompt
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

        critique = await self._perform_analysis_and_review(self.chapter_overview_prompt())
        critique = json.loads(critique)

        strengths = critique["strengths"]
        weaknesses = critique["weaknesses"]
        suggestions = critique["suggestions"]
        verses_to_review = critique["verses_to_review"]

        results.append(
            {
                "model": linguist.model_name,
                "agent_name": linguist.name,
                "chapter": int(verse_records[-1]["chapter"]) if verse_records else 0,
                "verse": 0,
                "biblical_text": verse_records[-1].get("biblical_text", "") if verse_records else "",
                "translation": verse_records[-1].get("translation_text", "") if verse_records else "",
                # "notes": row['notes'],
                "score": critique["score"],
                "model_analysis": f"# Strengths:\n{strengths}\n\n# Weaknesses:\n{weaknesses}\n\n# Suggestions:{suggestions}\n\n# Verses to Review:\n{verses_to_review}"
            } 
        )

        final_df = DataFrame(results)
        if output_csv_path is None:
            output_csv_path = Path("pragmatic_analysis.csv")
        else:
            output_csv_path = Path(output_csv_path)

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv_path, index=False)
        print("\n--- AutoGen Script Finished ---")
        print(f"Results saved to {output_csv_path}")
        print(final_df.to_markdown(index=False, numalign="left", stralign="left"))
        return final_df
