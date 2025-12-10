from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

import os 

key = os.getenv("OPENAI_API_KEY")
CLIENT_MODEL = "gpt-5-mini"

class CriticReview(BaseModel):
    """
    Structured output schema for the Linguistic Critic's review using a boolean.
    """
    accepted: bool = Field(..., description="True if the analysis is linguistically sound, False if revision is needed.")
    reasoning: str = Field(..., description="A Markdown formatted explanation for the decision, in English. If 'accepted' is False, this must contain the revision instructions.")

class CriticAgent:

    def __init__(self):
        self.name = "LINGUISTIC_CRITIC"
        self.system_message="""
        You are the Linguistic Critic. Your role is to rigorously review a submitted translation analysis.
        
        Your ONLY criteria for approval are that:
            1. The analysis must be based on verifiable linguistic, stylistic, or semantic arguments.
            2. Words and phrases being analyzed **MUST** be present in the texts being analyzed.
        
        You **MUST** output your response as a single JSON object with two fields: 'accepted' (boolean) and 'reasoning' (string).
        
        - If the critique is scientifically rigorous (linguistically sound), set "accepted" to **true**.
        - If the critique relies on popular commentary, personal opinion, or non-linguistic fields, set "accepted" to **false**, and use the "reasoning" field to provide clear revision instructions.
        
        DO NOT include any explanation or text outside of the JSON object.
        """
    
        self.client = OpenAIChatCompletionClient(
            api_type="openai",
            model=CLIENT_MODEL, 
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=key, 
            #seed=42, 
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "critic_review",
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
        