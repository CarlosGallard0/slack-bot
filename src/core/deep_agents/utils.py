from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")

load_dotenv()

def get_embeddings_from_provider():
    """Get embeddings client based on configured provider."""
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    
    if provider == "vertexai":
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION", "us-west1")
        api_key = os.getenv("VERTEX_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-005",
            project=project_id,
            location=location,
            api_key=api_key
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return embeddings

def async_to_sync(async_func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Thread-safe decorator to convert async function to sync.
    Works in Jupyter notebooks and ThreadPoolExecutor contexts.
    """

    @wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            # Try to get the running loop (if we're in async context)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - we're in a sync context or thread pool
            loop = None

        if loop is not None and loop.is_running():
            # We're in an async context with a running loop
            # This shouldn't happen with LangGraph subagents, but just in case
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(async_func(*args, **kwargs))
        else:
            # We're in a sync context or thread pool - create new event loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

    return wrapper

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