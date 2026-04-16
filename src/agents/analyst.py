from pydantic import BaseModel, ConfigDict, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

class PragmaticAnalystReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    """
    Structured output schema for the Linguistic Critic's review using a boolean.
    """
    score: int = Field(..., description="A score from 1 to 10 which indicates how well the meaning of the original language is preserved in the translation, where 1 means that the meaning is substantially different and 10 means that all in line and between the lines meaning is preserved in the translation.")
    confidence: int = Field(..., ge=0, le=100, description="The analyst's confidence in this evaluation from 0 to 100.")
    reasoning: str = Field(..., description="A Markdown formatted explanation of the pragmatic analysis of the original text and how the target translation handles those dynamics.")
    verses_to_review: list[int] = Field(..., description="A list of the verses within the chapter that need the most attention for revision by the translator. Leave this field empty unless prompted to do a final chapter overview.")
    strengths: str = Field(..., description="A Markdown formatted list (in English) of things the translation does well with respect to the annotations.")
    weaknesses: str = Field(..., description="A Markdown formatted list (in English) of things the translation does not do well with respect to the annotations.")
    suggestions: str = Field(..., description="A Markdown formatted (in English) list of concrete ways the translation could better reflect the pragmatic meaning.")


class PragmaticAnalystAgent:

    task_description = f"""Evaluate the following Bible translation with respect to the pragmatic goal, focusing primarily on whether the implicit meaning—that is, what is communicated between the lines, in addition to what is stated directly—is preserved in the translation. 
    Your rationale must be based on the provided expert annotations and context from previous verses as the analysis moves verse by verse through a chapter.\n
    Additionally, you will score the translation (from 1 to 10) based on how well it retains the meaning of the original text.\n
    You must also provide a confidence score from 0 to 100 indicating how confident you are in the analysis, and a Markdown-formatted explanation describing the pragmatic analysis of the original text and how the target translation handles those dynamics.\n
    If you are prompted for a final chapter overview, submit a list of verses that need the most review and use strengths, weaknesses, and suggestions for feedback on the entire chapter.
    """

    def __init__(self, name, model_name, api_key, base_url, response_format):
        self.name = name
        self.model_name = model_name
        self.system_message =f"ROLE: You are performing a cross-lingual pragmatic analysis of a biblical translation, where the original text is either in Ancient Hebrew or Koine Greek.\n" + self.task_description

        
    
        self.client = OpenAIChatCompletionClient(
            api_type="openai",
            model=model_name, 
            base_url=base_url,
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=api_key,
            timeout=60,
            response_format=response_format
        )

        self.agent = AssistantAgent(
            name=self.name,
            system_message=self.system_message,
            model_client=self.client
        )

    def get_agent(self):
        return self.agent    
