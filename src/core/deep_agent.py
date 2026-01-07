from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Any, Optional, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio
from src.core.graphiti import GraphitiClient

load_dotenv()

graph_client = GraphitiClient()

class GraphSearchResult(BaseModel):
    """Knowledge graph search result model."""

    fact: str
    uuid: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    source_node_uuid: Optional[str] = None


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


# Define the subagent for graph search
GRAPH_SEARCH_SUBAGENT = {
    "name": "graph_search",
    "description": "Searches the knowledge graph to find facts, relationships, and temporal information about entities.",
    "system_prompt": """You are a knowledge graph search specialist. Your role is to:
1. Take user queries about facts and relationships
2. Search the knowledge graph using the graph_search tool
3. Return comprehensive results with facts and their relationships

Always use the graph_search tool to find relevant information in the knowledge graph.""",
}

SUBAGENT_SELECTION_GUIDE = """You are a helpful research assistant coordinator. Route user queries to the appropriate subagent:

- If the user asks about facts, timelines, relationships, or any information that should be searched in the knowledge graph, route to the graph_search subagent.
- The graph_search subagent has access to a comprehensive knowledge graph and can find detailed information.

Always delegate to the graph_search subagent for knowledge-based queries."""

class AgentCore:
    def __init__(self):
        self.llm = get_model_from_provider()
        self.agent_executor = None

    def setup(self):
        """
        Sets up the Agent with graph search capability using subagents.
        """

        @tool
        async def graph_search(query: str) -> Any:
            """
            Search the knowledge graph for facts and relationships.

            This tool queries the knowledge graph to find specific facts, relationships
            between entities, and temporal information.

            Args:
                query: Search query to find facts and relationships

            Returns:
                Search results as formatted text
            """
            try:
                # Await the async graph search on the current event loop
                search_results = await graph_client.search(query)
            except Exception as e:
                return f"Graph search error: {str(e)}"
            
            return [
            GraphSearchResult(
                fact=r["fact"],
                uuid=r["uuid"],
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid"),
            )
            for r in search_results]

        # Create the graph_search subagent
        graph_search_subagent = {
            "name": "graph_search",
            "description": "Searches the knowledge graph to find facts, relationships, and temporal information about entities.",
            "system_prompt": """You are a knowledge graph search specialist. Your role is to:
1. Take user queries about facts and relationships
2. Search the knowledge graph using the graph_search tool
3. Return comprehensive results with facts and their relationships

Always use the graph_search tool to find relevant information in the knowledge graph.""",
            "tools": [graph_search],
        }

        subagents = [graph_search_subagent]

        system_prompt = """You are a helpful research assistant coordinator. Route user queries to the appropriate subagent:

- If the user asks about facts, timelines, relationships, or any information that should be searched in the knowledge graph, route to the graph_search subagent.
- The graph_search subagent has access to a comprehensive knowledge graph and can find detailed information.

Always delegate to the graph_search subagent for knowledge-based queries."""

        checkpointer = MemorySaver()

        self.agent_executor = create_deep_agent(
            model=self.llm,
            system_prompt=system_prompt,
            subagents=subagents,
            checkpointer=checkpointer,
        )

    def ask(self, query: str) -> str:
        """
        Processes a query through the RAG agent.
        """
        if not self.agent_executor:
            self.setup()
        # If there's no running event loop we can run the async path synchronously.
        # In async contexts (like notebooks) we attempt to run the coroutine
        # on the existing loop using `nest_asyncio`. If `nest_asyncio` is not
        # available, instruct the user to `await agent.ask_async(...)`.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to run the async method
            return asyncio.run(self.ask_async(query))
        else:
            # Event loop already running — try to run the coroutine on the
            # same loop using nest_asyncio so notebooks continue to work.
            try:
                import nest_asyncio
            except Exception as e:
                raise RuntimeError(
                    "Event loop already running; install and import `nest_asyncio`, or use `await agent.ask_async(...)` in async contexts."
                ) from e

            # Patch the running loop to allow nested run_until_complete
            nest_asyncio.apply()
            return loop.run_until_complete(self.ask_async(query))

    async def ask_async(self, query: str) -> str:
        """
        Async version of ask() — call this with `await agent.ask_async(...)`
        """
        if not self.agent_executor:
            self.setup()

        config = {"configurable": {"thread_id": "default_thread"}}
        # Await the agent/graph invoke directly (keeps everything in same loop)
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
