from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.deep_agents.graph_search_agent import graph_search_subagent
from dotenv import load_dotenv
from src.core.deep_agents.utils import GRAPH_SEARCH_SUBAGENT, SUBAGENT_SELECTION_GUIDE, normalize_agent_response
import os
import asyncio
import nest_asyncio

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
        self.memory = MemorySaver()
        self.llm = None
        self.agent_executor = None
        self._loop = None

    def setup(self):
        """
        Sets up the Agent with graph search capability using subagents.
        """
        self.llm = get_model_from_provider()

        subagents = [graph_search_subagent]

        self.agent_executor = create_deep_agent(
            model=self.llm,
            system_prompt=SUBAGENT_SELECTION_GUIDE,
            subagents=subagents,
            checkpointer=self.memory,
        )
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None


    def ask(self, query: str, thread_id: str = "default_thread") -> str:
        """
        Processes a query through the RAG agent.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(query, thread_id))
        else:
            try:
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Event loop already running; install `nest_asyncio` (pip install nest_asyncio), or use `await agent.ask_async(...)` in async contexts."
                )
            
            return loop.run_until_complete(self.ask_async(query, thread_id))

    async def ask_async(self, query: str, thread_id: str = "default_thread") -> str:
        """
        Async version of ask() — always returns a string
        """
        current_loop = asyncio.get_running_loop()
        if self._loop is not None and current_loop is not self._loop:
            self.agent_executor = None

        if not self.agent_executor:
            self.setup()

        if isinstance(query, dict):
            query = query.get("content") or query.get("text") or query.get("query") or str(query)
        query = str(query).strip()

        config = {"configurable": {"thread_id": thread_id}}
        response = await self.agent_executor.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )

        return normalize_agent_response(response)
