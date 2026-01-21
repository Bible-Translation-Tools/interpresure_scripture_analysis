from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

class SecretaryAgent:

    def __init__(self, name, model_name, api_key, base_url, task_description="", response_format="text"):
        self.name = name
        self.model_name = model_name
        self.system_message =f"You are {name}, a secretary summarizing a linguistics debate." + task_description
    
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
    
    async def summarize(self, opening_statements, debate_transcript, closing_statements):
        task = self._construct_prompt(opening_statements, debate_transcript, closing_statements)
        summary = await self.agent.run(task=task)
        return summary.messages[-1].content

    def _construct_prompt(self, opening_statements, debate_transcript, closing_statements):
        prompt = (
            f"ROLE: You are summarizing a linguistic debate.\n"
            "---------------------------------------------------------------------------------\n"
            f"Opening Statements:\n"
            f"{opening_statements}\n\n"
            f"Debate:\n"
            f"{debate_transcript}\n\n"
            f"Closing Statements:\n"
            f"{closing_statements}\n\n\n"
            "---------------------------------------------------------------------------------\n"
            f"TASK:"
            "Provide a summary of the debate, extracting the final conclusions. "
            "Focus on the specific strengths of the translation being debated, and the specific weaknesses. "
            "If improvement suggestions were made, list exactly what should be done."
            "If the interlocutors did not reach a consensus, list the specific reasons why they disagreed."
            "Readers of your summary MUST be able to take the feedback DIRECTLY from your summary WITHOUT needing to refer back to the debate itself- it is critical that no important details are left out."
            "Format your summary in Markdown."
        )
        
        return prompt
        