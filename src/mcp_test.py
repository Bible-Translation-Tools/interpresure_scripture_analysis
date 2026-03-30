import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import os


load_dotenv()
MACULA_DB_PATH = "macula_greek_nestle1904.db"
PRAGMATICS_DB_PATH = "interpresure_phm.db"
BART_DB_PATH = "NT_BART_Annotations.db"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    print("--- Bible Verse Translation Analyzer ---")
    book = "phm".upper() 
    chapter = 1 
    verse = 6 
    user_translation = "I pray that the sharing of your faith may be effective, so you will have a full understanding of every good thing we have in Christ." # "Tôi cầu xin rằng sự thông công của đức tin anh có ích cho sự hiểu biết về mọi việc lành đang ở giữa chúng ta trong Chúa Cứu Thế." 
    user_greek = "ὅπως ἡ κοινωνία τῆς πίστεώς σου ἐνεργὴς γένηται ἐν ἐπιγνώσει παντὸς ἀγαθοῦ τοῦ ἐν ἡμῖν εἰς Χριστόν." 

    macula_server = StdioServerParams(
        command="mcp-server-sqlite-npx",
        args=[MACULA_DB_PATH]
    )
    pragmatics_server = StdioServerParams(
        command="mcp-server-sqlite-npx",
        args=[PRAGMATICS_DB_PATH]
    )

    bart_server = StdioServerParams(
        command="mcp-server-sqlite-npx",
        args=[BART_DB_PATH]
    )

    model_client = OpenAIChatCompletionClient(
        api_type="openai",
        model="gpt-5",
        # model_info=ModelInfo(vision=True, function_calling=True), 
        api_key=OPENAI_KEY,
    )
    
    macula_tools = await mcp_server_tools(macula_server)
    pragmatics_tools = await mcp_server_tools(pragmatics_server)
    bart_tools = await mcp_server_tools(bart_server)


    macula_agent = AssistantAgent(
        name="macula_expert",
        model_client=model_client,
        tools=macula_tools,
        system_message="""
        You extract Greek morphological and syntactic data for a specific verse. 
        After every tool use, explicitly describe:
            - Which tool was used
            - The exact parameters passed
            - The key results returned""",
        reflect_on_tool_use=True
    )

    discourse_agent = AssistantAgent(
        name="discourse_expert",
        model_client=model_client,
        tools=bart_tools,
        system_message="""
        You extract linguistic discourse data for a specific verse. 
        After every tool use, explicitly describe:
            - Which tool was used
            - The exact parameters passed
            - The key results returned""",
        reflect_on_tool_use=True
    )

    pragmatics_agent = AssistantAgent(
        name="pragmatics_expert",
        model_client=model_client,
        tools=pragmatics_tools,
        system_message="""
        You extract pragmatic and discourse annotations for a specific verse.
        After every tool use, explicitly describe:
            - Which tool was used
            - The exact parameters passed
            - The key results returned""",
        reflect_on_tool_use=True
    )

    lead_analyst = AssistantAgent(
        name="lead_analyst",
        model_client=model_client,
        system_message="""
            You are the lead scholar. You receive data from macula_expert, discourse_expert, and pragmatics_expert on the verse provided by the user. Then, compare that data
            to the user's translation and evaluate the quality of the translation based on the data. Conclude with a summary of strengths and weaknesses of the translation with respect to the data and provide suggestions for improvement.
            Format your report in Markdown.
        """
    )

    summary_agent = AssistantAgent(
        name="summary_agent",
        model_client=model_client,
        system_message="You are a linguistics teacher. You will review a linguistic analysis and communicate it in simple English that is easy for a lay person to understand. Format your report in Markdown."
    )

    team = RoundRobinGroupChat([macula_agent, discourse_agent, pragmatics_agent, lead_analyst, summary_agent], max_turns=6)
    
    task = (
        f"Analyze {book} {chapter}:{verse}.\n"
        f"Greek: {user_greek}\n"
        f"Translation: {user_translation}"
    )

    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())