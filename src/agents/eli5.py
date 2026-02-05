from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

class Eli5Response(BaseModel):
    """The structured output for a simplified summary."""
    agent_name: str = Field(description="The name of the speaker.")
    summary: str = Field(description="Your Markdown formatted summary, in simple English.")

class Eli5Agent:

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "linguist_review",
            "schema": Eli5Response.model_json_schema()
        }
    }

    def __init__(self, name, model_name, api_key, base_url, task_description="", response_format="text"):
        self.name = name
        self.model_name = model_name
        self.system_message =f"You are PROFESSOR_P_MATICS, a linguistics teacher and expert in communicating linguistic concepts in simple terms for lay people to understand." 
    
        self.client = OpenAIChatCompletionClient(
            api_type="openai",
            model=model_name, 
            base_url=base_url,
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
            api_key=api_key,
            timeout=60,
            response_format=self.response_format
        )

        self.agent = AssistantAgent(
            name=self.name,
            system_message=self.system_message,
            model_client=self.client
        )

    def get_agent(self):
        return self.agent
    
    async def eli5(self, summary):
        task = self._construct_prompt(summary)
        summary = await self.agent.run(task=task)
        return summary.messages[-1].content

    def _construct_prompt(self, summary):
        prompt = (
            f"ROLE: You are communicating a linguistic analysis report in plain English so that even a middle school student could understand what was said.\n"
            "---------------------------------------------------------------------------------\n"
            f"{summary}"
            "---------------------------------------------------------------------------------\n"
            f"TASK:"
            "Communicate everything that was said in the analysis in simple terms."
            "Format your summary in Markdown."
        )
        
        return prompt
        