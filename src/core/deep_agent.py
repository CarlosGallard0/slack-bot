from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.deep_agents.graph_search_agent import graph_search_subagent
from dotenv import load_dotenv
from src.core.deep_agents.utils import GRAPH_SEARCH_SUBAGENT, SUBAGENT_SELECTION_GUIDE
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
            api_key=api_key,
            thinking_budget=512,
            include_thoughts=True,
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
        # Always get a fresh LLM instance to ensure it binds to the current loop/context
        self.llm = get_model_from_provider()

        subagents = [graph_search_subagent]

        self.agent_executor = create_deep_agent(
            model=self.llm,
            system_prompt=SUBAGENT_SELECTION_GUIDE,
            subagents=subagents,
            checkpointer=self.memory,
        )
        # Track the loop this executor was created on
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _normalize_agent_response(self, response) -> str:
        """
        Normalize LangChain / LangGraph agent responses into plain text.
        """

        # Case 1: LangGraph dict with messages
        if isinstance(response, dict):
            if "messages" in response and response["messages"]:
                return self._normalize_agent_response(response["messages"][-1])

            if "output" in response:
                return str(response["output"]).strip()

        # Case 2: Message object with .content
        content = getattr(response, "content", None)
        if content is not None:
            return self._normalize_agent_response(content)

        # Case 3: List of content blocks (YOUR BUG)
        if isinstance(response, list):
            parts = []
            for item in response:
                if isinstance(item, dict):
                    # Standard text block
                    if item.get("type") == "text" and "text" in item:
                        parts.append(item["text"])
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()

        # Case 4: Plain string
        if isinstance(response, str):
            return response.strip()

        # Final fallback
        return str(response).strip()

    def ask(self, query: str) -> str:
        """
        Processes a query through the RAG agent.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.ask_async(query))
        else:
            # Event loop already running - use nest_asyncio to allow nested event loops
            try:
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Event loop already running; install `nest_asyncio` (pip install nest_asyncio), or use `await agent.ask_async(...)` in async contexts."
                )

            # Run the async function in the current event loop
            return loop.run_until_complete(self.ask_async(query))

    async def ask_async(self, query: str) -> str:
        """
        Async version of ask() — always returns a string
        """
        # Check if we need to re-initialize due to loop change
        current_loop = asyncio.get_running_loop()
        if self._loop is not None and current_loop is not self._loop:
            # Loop changed! We must recreate the executor to bind to the new loop
            self.agent_executor = None

        if not self.agent_executor:
            self.setup()

        # Normalize input
        if isinstance(query, dict):
            query = (
                query.get("content")
                or query.get("text")
                or query.get("query")
                or str(query)
            )
        query = str(query).strip()

        config = {"configurable": {"thread_id": "default_thread"}}
        response = await self.agent_executor.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )

        return self._normalize_agent_response(response)
