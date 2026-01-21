import os
import asyncio
from typing import List, Dict, Any, Optional
from src.rag.graphiti import GraphitiClient
from src.rag.raptor import RaptorLangChain

class RAGAdapter:
    """Adapter to unify Graphiti and Raptor RAG systems."""
    
    def __init__(self, rag_type: str = "graphiti"):
        self.rag_type = rag_type.lower()
        self.graphiti_client: Optional[GraphitiClient] = None
        self.raptor_clients: Dict[str, RaptorLangChain] = {}
        
    async def initialize(self, index_id: str = "default"):
        """Initialize the RAG client if needed."""
        if self.rag_type == "graphiti":
            if not self.graphiti_client:
                self.graphiti_client = GraphitiClient()
            await self.graphiti_client.initialize()
        elif self.rag_type == "raptor":
            if index_id not in self.raptor_clients:
                # Try to load existing, otherwise create new
                client = RaptorLangChain.from_existing(collection_name=index_id)
                if not client:
                    client = RaptorLangChain(collection_name=index_id)
                self.raptor_clients[index_id] = client
            # Raptor checkpints are loaded during init
        else:
            raise ValueError(f"Unknown RAG type: {self.rag_type}")

    async def search(self, query: str, index_id: str = "default") -> List[Dict[str, Any]]:
        """Search the selected RAG system and return results in a unified format."""
        if self.rag_type == "graphiti":
            if not self.graphiti_client:
                await self.initialize()
            
            edges = await self.graphiti_client.search(query)
            return [
                {
                    "text": edge["fact"],
                    "source": edge.get("source_title", "Graphiti Knowledge Graph"),
                    "score": 0.5,  # Graphiti doesn't provide a direct score in this wrapper yet
                    "metadata": edge
                }
                for edge in edges
            ]
        elif self.rag_type == "raptor":
            if index_id not in self.raptor_clients:
                await self.initialize(index_id)
            
            client = self.raptor_clients[index_id]
            # Raptor retrieve is synchronous, run in thread to avoid blocking loop
            results = await asyncio.to_thread(client.retrieve, query, top_k=5)
            
            return [
                {
                    "text": r["text"],
                    "source": r.get("source_doc", "Raptor RAG"),
                    "score": r.get("score", 0.5),
                    "metadata": r.get("metadata", {})
                }
                for r in results
            ]
        else:
            raise ValueError(f"Unknown RAG type: {self.rag_type}")
