from langchain_core.tools import tool
from typing import Any, Optional
from src.core.deep_agents.graphiti import GraphitiClient
from pydantic import BaseModel

graph_client = GraphitiClient()

class GraphSearchResult(BaseModel):
    """Knowledge graph search result model."""

    fact: str
    uuid: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    source_node_uuid: Optional[str] = None

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