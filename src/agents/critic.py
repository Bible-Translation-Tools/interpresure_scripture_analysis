from pydantic import BaseModel, ConfigDict, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from model.config import get_config_for_model

class CriticReview(BaseModel):
    model_config = ConfigDict(extra='forbid')

    """
    Structured output schema for the Linguistic Critic's review using a boolean.
    """
    accepted: bool = Field(..., description="True if the analysis is linguistically sound, False if revision is needed.")
    reasoning: str = Field(..., description="A Markdown formatted explanation for the decision, in English. If 'accepted' is False, this must contain the revision instructions.")

class CriticAgent:

    def __init__(self, critic_model, biblical_language="greek", api_key=None, base_url=None):
        self.name = "LINGUISTIC_CRITIC"
        config = get_config_for_model(critic_model)
        resolved_api_key = api_key if api_key is not None else config.get("key")
        resolved_base_url = base_url if base_url is not None else config.get("base_url")
        self.system_message=f"""
        You are the Linguistic Critic. Your role is to rigorously review a submitted translation analysis.
        
        Your ONLY criteria for approval are that:
            1. The analysis must be based on verifiable linguistic, pragmatic, or semantic arguments.
            2. Words and phrases being analyzed **MUST** be present in the texts being analyzed.
            3. The analysis must be limited **ONLY** to how the translation handles the original text with respect to the provided pragmatic annotations.
        
        You **MUST** output your response as a single JSON object with two fields: 'accepted' (boolean) and 'reasoning' (string).
        
        - If the critique is scientifically rigorous (linguistically sound), set "accepted" to **true**.
        - If the critique relies on popular commentary, personal opinion, or non-linguistic fields, set "accepted" to **false**, and use the "reasoning" field to provide clear revision instructions.
        
        DO NOT include any explanation or text outside of the JSON object.
        """
    
        self.client = OpenAIChatCompletionClient(
            api_type="openai",
            model=config["model"], 
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=resolved_api_key, 
            base_url=resolved_base_url,
            #seed=42, 
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "critic_review",
                    "strict": True,
                    "schema": CriticReview.model_json_schema()
                }
            }
        )

        self.agent = AssistantAgent(
            name=self.name,
            system_message=self.system_message,
            model_client=self.client
        )

    def get_agent(self):
        return self.agent
        
