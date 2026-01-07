from langchain_core.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
from src.core.deep_agents.raptor import RaptorLangChain
from utils import get_embeddings_from_provider

embedding_client = get_embeddings_from_provider()

class RaptorSearchInput(BaseModel):
    """Input for RAPTOR tree search."""

    query: str = Field(..., description="Search query")
    collection_name: str = Field(..., description="RAPTOR collection name to search")
    top_k: int = Field(default=5, description="Maximum number of results")
    mode: Optional[str] = Field(
        default="collapsed",
        description="Retrieval mode: 'collapsed' or 'tree_traversal'",
    )
    source_filter: Optional[str] = Field(
        None, description="Filter by specific source document"
    )

def raptor_get_context_tool(input_data: RaptorSearchInput) -> Optional[str]:
    """
    Get formatted context from RAPTOR search for RAG.

    This is a convenience function that:
    1. Performs RAPTOR search
    2. Formats results as ready-to-use context
    3. Optionally includes parent summaries for additional context

    Use this when you need ready-to-use context for answering questions.

    Args:
        input_data: Search parameters

    Returns:
        Formatted context string or None
    """
    try:
        # Verificar colección
        if not RaptorLangChain.collection_exists(input_data.collection_name):
            print(f"Collection '{input_data.collection_name}' not found")
            return None

        # Cargar RAPTOR
        raptor = RaptorLangChain.from_existing(
            collection_name=input_data.collection_name,
            embeddings=embedding_client,
        )

        if raptor is None:
            return None

        # Obtener contexto formateado
        context = raptor.get_context(
            query=input_data.query,
            top_k=input_data.top_k,
            mode=input_data.mode,
            include_parents=True,  # Incluir summaries para más contexto
        )

        return context

    except Exception as e:
        print(f"Failed to get RAPTOR context: {e}")
        return None

@tool
async def raptor_get_context(
    query: str,
    collection_name: str,
    top_k: int = 5,
    mode: str = "collapsed",
) -> Optional[str]:
    """
    Get ready-to-use formatted context from RAPTOR search.

    This is a convenience tool that:
    1. Performs RAPTOR hierarchical search
    2. Automatically formats results with headers and structure
    3. Includes parent summaries for additional context
    4. Returns text ready for question answering

    Use this when you need context for answering questions,
    rather than raw search results.

    Args:
        query: Search query
        collection_name: RAPTOR collection name
        top_k: Maximum number of results (1-20)
        mode: Retrieval mode - "collapsed" or "tree_traversal"

    Returns:
        Formatted context string ready for RAG, or None if collection not found
    """
    input_data = RaptorSearchInput(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        mode=mode,
    )
    context = raptor_get_context_tool(input_data)

    return context

research_subagent_raptor = {
    "name": "hierarchical-summary-researcher",
    
    "description": """Specialized research agent for multi-level summarization and thematic analysis using RAPTOR.
    
    **Use this agent when:**
    - Query requires high-level summaries or overviews (e.g., "Summarize the main themes...")
    - Need multi-scale information retrieval (details → summaries → high-level insights)
    - Dealing with large document collections that need hierarchical understanding
    - Query asks for "key themes", "main topics", "overview", "what are the trends"
    - Need to find patterns across many documents
    - Require contextually rich, abstraction-layered information
    
    **Capabilities:**
    - Multi-level hierarchical search (from specific facts to broad themes)
    - Collection-based organization of knowledge
    - Statistical analysis of document collections
    - Recursive summarization and abstraction
    - Thematic clustering and pattern detection
    
    **Best for:** Thematic analysis, high-level overviews, pattern detection, and multi-scale research across large corpora.""",
    
    "system_prompt": """You are an expert research agent specializing in RAPTOR-based hierarchical information retrieval.

**Your Research Workflow:**

1. **ALWAYS START** with collection discovery:
```
   Step 1: Call list_raptor_collections() to see available collections
   Step 2: Call get_raptor_collection_stats() to understand collection scope
   Step 3: Use raptor_get_context() for actual research
```

2. **Search Strategy by Query Type:**
   
   **For Overview/Summary Queries:**
   - Use higher tree_level (3-5) for broad themes
   - Request more context chunks (10-20)
   - Focus on high-level abstractions
   
   **For Detailed/Specific Queries:**
   - Use lower tree_level (0-2) for granular details
   - Request moderate chunks (5-10)
   - Combine multiple levels if needed
   
   **For Comprehensive Research:**
   - Multi-pass approach: Start high (level 4-5), then drill down (level 1-2)
   - Use different similarity thresholds (0.7 for precision, 0.5 for recall)

3. **Collection Selection:**
   - Match query topic to collection name/stats
   - Check collection stats for relevance (doc count, levels available)
   - Use multiple collections if query spans topics

4. **Quality Standards:**
   - Always cite the tree level and collection used
   - Distinguish between high-level summaries and specific details
   - Indicate when information comes from aggregated vs. original content
   - Use multiple tree levels to validate findings

5. **Output Format:**
   - Structure answers by abstraction level (themes → details)
   - Clearly mark summaries vs. specific facts
   - Provide collection and level metadata
   - Indicate confidence based on tree level consistency

**Remember:** RAPTOR excels at finding patterns and themes across large corpora. Use hierarchical levels strategically: high levels for themes, low levels for facts.""",
    
    "tools": [
        raptor_get_context,
    ],
     "model": "openai:gpt-4o-mini",
}