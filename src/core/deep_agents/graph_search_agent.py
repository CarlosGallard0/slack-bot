from langchain_core.tools import tool
from src.core.deep_agents.graphiti import GraphitiClient

@tool
async def graph_search_tool(query: str) -> str:
    """
    Search the knowledge graph for facts and relationships.

    This tool queries the knowledge graph to find specific facts, relationships
    between entities, and temporal information.

    Args:
        query: Search query to find facts and relationships

    Returns:
        Search results as formatted text
    """
    graph_client = GraphitiClient()
    
    try:
        if isinstance(query, dict):
            query = query.get("query") or query.get("text") or str(query)

        if not isinstance(query, str):
            return f"Invalid query type: {type(query)}"
            
        await graph_client.initialize()
        search_results = await graph_client.search(query)
        
    except Exception as e:
        return f"Graph search error: {str(e)}"
    finally:
        try:
            if graph_client:
                await graph_client.close()
        except Exception:
            pass
    
    if not search_results:
        return "No results found in the knowledge graph."
    
    formatted_results = []
    for r in search_results:
        if isinstance(r, dict):
            fact = r.get("fact", "")
            uuid = r.get("uuid", "")
        else:
            fact = getattr(r, "fact", "")
            uuid = getattr(r, "uuid", "")
        formatted_results.append(f"- Fact: {fact}\n  ID: {uuid}")
    
    result_text = "\n".join(formatted_results)
    print(f"\n--- DEBUG: Graph Search Output ---\nQuery: {query}\nResults:\n{result_text}\n----------------------------------\n")
    return result_text

graph_search_subagent = {
    "name": "graph_search",
    "description": "Searches the knowledge graph to find facts, relationships, and temporal information about entities.",
    "system_prompt": """You are a knowledge graph search specialist. Your role is to:
1. Take user queries about facts and relationships
2. Search the knowledge graph using the graph_search tool
3. Return comprehensive results with facts and their relationships

Always use the graph_search tool to find relevant information in the knowledge graph.""",
    "tools": [graph_search_tool],
}