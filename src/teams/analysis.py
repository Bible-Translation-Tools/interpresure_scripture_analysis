import json
from autogen_agentchat.teams import RoundRobinGroupChat
from agents.critic import CriticReview


def parse_critic_output(json_str: str) -> CriticReview | None:
    """Parses the Critic's JSON output."""
    try:
        data = json.loads(json_str.strip())
        return CriticReview(**data)
    except Exception as e:
        print(f"Error parsing Critic JSON (Attempted to parse: '{json_str.strip()[:50]}...'): {e}")
        return None


class LinguisticAnalysis:
    def __init__(self,
                 llm_config,
                 linguist_agent,
                 critic_agent):
        self.llm_config = llm_config
        self.linguist_agent = linguist_agent
        self.critic_agent = critic_agent

    async def perform_analysis_and_review(self, analysis_prompt):
        """
        Uses a single RoundRobinGroupChat, driving it by calling run(task=...) for each
        explicit user-directed turn (linguist -> critic -> linguist ...).
        """

        # create one persistent team (keep max_turns small so each run returns quickly)
        chat = RoundRobinGroupChat(
            participants=[self.linguist_agent, self.critic_agent],
            max_turns=1  # one agent turn per run()
        )

        print(f"\n--- Starting Independent Analysis for {self.linguist_agent.name} ---")

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
                f"{self.critic_agent.name}: Review the following analysis and respond ONLY in JSON.\n\n"
                f"{CriticReview.schema_json()}\n\n"
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
                f"{self.linguist_agent.name}: Your previous analysis was rejected because:\n"
                f"'{critic_review.reasoning}'.\n"
                "Please revise your critique to ensure it is based solely on verifiable linguistic principles. "
                "Retain the 'Score: [N]' format."
            )

            revision_result = await chat.run(task=revision_prompt)
            critique_for_review = revision_result.messages[-1].content
            review_round += 1

        print("⚠️ Max review rounds reached — returning last version.")
        return critique_for_review
