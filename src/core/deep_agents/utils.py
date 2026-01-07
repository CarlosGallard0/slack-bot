from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

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