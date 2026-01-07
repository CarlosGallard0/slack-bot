from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.deep_agents.graph_search_agent import graph_search_subagent
from dotenv import load_dotenv
from src.core.deep_agents.utils import GRAPH_SEARCH_SUBAGENT, SUBAGENT_SELECTION_GUIDE
import os
import asyncio

load_dotenv()

def get_model_from_provider():
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
    llm = None

    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "vertexai":
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION", "us-west1")
        api_key = os.getenv("VERTEX_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            project=project_id,
            location=location,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return llm


class AgentCore:
    def __init__(self):
        self.llm = get_model_from_provider()
        self.agent_executor = None

    def setup(self):
        """
        Sets up the Agent with graph search capability using subagents.
        """

        subagents = [graph_search_subagent]

        checkpointer = MemorySaver()

        self.agent_executor = create_deep_agent(
            model=self.llm,
            system_prompt=SUBAGENT_SELECTION_GUIDE,
            subagents=subagents,
            checkpointer=checkpointer,
        )

    def ask(self, query: str) -> str:
        """
        Processes a query through the RAG agent.
        """
        if not self.agent_executor:
            self.setup()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(query))
        else:
            try:
                import nest_asyncio
            except Exception as e:
                raise RuntimeError(
                    "Event loop already running; install and import `nest_asyncio`, or use `await agent.ask_async(...)` in async contexts."
                ) from e

            nest_asyncio.apply()
            return loop.run_until_complete(self.ask_async(query))

    async def ask_async(self, query: str) -> str:
        """
        Async version of ask() — call this with `await agent.ask_async(...)`
        """
        if not self.agent_executor:
            self.setup()

        config = {"configurable": {"thread_id": "default_thread"}}
        response = await self.agent_executor.ainvoke({"messages": [{"role": "user", "content": query}]}, config=config)

        # # Normalize response
        # if isinstance(response, dict):
        #     if "messages" in response and response["messages"]:
        #         last = response["messages"][-1]
        #         if isinstance(last, dict):
        #             return last.get("content") or str(last)
        #         # if it's an object with .content:
        #         return getattr(last, "content", str(last))
        #     if "output" in response:
        #         return response["output"]
        return response
