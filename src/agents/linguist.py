from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

class LinguistReview(BaseModel):
    """
    Structured output schema for the Linguistic Critic's review using a boolean.
    """
    score: int = Field(..., description="The score of the translation (1-10, where 10 is best).")
    reasoning: str = Field(..., description="A Markdown formatted explanation for the decision, in English.")


class LinguistAgent:

    def __init__(self, name, model_name, api_key, base_url, task_description, response_format):
        self.name = name
        self.model_name = model_name
        self.system_message =f"You are {name}, a descriptive linguist and cross-cultural pragmatics expert." + task_description

        
    
        self.client = OpenAIChatCompletionClient(
            api_type="openai",
            model=model_name, 
            base_url=base_url,
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=api_key,
            response_format=response_format
        )

        self.agent = AssistantAgent(
            name=self.name,
            system_message=self.system_message,
            model_client=self.client
        )

    def get_agent(self):
        return self.agent

    def _construct_prompt(self, chapter, verse, greek, pragmatic_goal, notes, translation):
        prompt = (
            f"ROLE: You are performing a cross-lingual pragmatic analysis.\n"
            "---------------------------------------------------------------------------------\n"
            f"GROUND TRUTH ANALYSIS:\n"
            f"1. Context: Philemon {chapter}:{verse}\n"
            f"2. Greek Text: \"{greek}\"\n"
            f"3. Pragmatic Goal ('Face'): \"{pragmatic_goal}\"\n"
            f"4. Expert Rationale (The Constraint): \"{notes}\"\n\n"
            "TASK:\n"
            f"Evaluate the following translation against the Greek pragmatic goal and rationale.\n"
            f"Translation: \"{translation}\"\n\n"
            "FIRST, assign a score (1-10) to the translation for its fidelity to the 'Face' goal. "
            "SECOND, justify your score in English by identifying the specific linguistic feature (e.g., verb tense, politeness markers) that either succeeds or fails. Provide the justification in Markdown format."
        )
        
        return prompt
        