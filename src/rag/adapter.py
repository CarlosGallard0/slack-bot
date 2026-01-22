import asyncio
from typing import List, Dict, Any, Optional

from src.rag.graphiti import GraphitiClient
from src.rag.raptor import RaptorClient


class RAGAdapter:
    def __init__(self, rag_type: str = "graphiti"):
        self.rag_type = rag_type.lower()

        self.graphiti_client: Optional[GraphitiClient] = None
        self.raptor_client: Optional[RaptorClient] = None

    async def initialize(self, index_id: str = None):
        if self.rag_type == "graphiti":
            if not self.graphiti_client:
                print("🧠 [RAGAdapter] Initializing GRAPHITI client")
                self.graphiti_client = GraphitiClient()
                await self.graphiti_client.initialize()

        elif self.rag_type == "raptor":
            if not self.raptor_client:
                print("🧠 [RAGAdapter] Initializing RAPTOR client")
                self.raptor_client = RaptorClient()

        else:
            raise ValueError(f"Unknown RAG type: {self.rag_type}")

    async def search(self, query: str, index_id: str = None) -> List[Dict[str, Any]]:

        print(f"🧠 [RAGAdapter] Executing {self.rag_type.upper()} search")

        if self.rag_type == "graphiti":
            if not self.graphiti_client:
                await self.initialize()

            edges = await self.graphiti_client.search(query)

            return [
                {
                    "text": edge.get("fact", ""),
                    "source": edge.get("source_title", "Graphiti Knowledge Graph"),
                    "score": 0.5,
                    "metadata": edge,
                }
                for edge in edges
            ]
        elif self.rag_type == "raptor":
            if not self.raptor_client:
                await self.initialize()

            results = await asyncio.to_thread(
                self.raptor_client.search,
                query,
                5,
            )

            return [
                {
                    "text": r["text"],
                    "source": r["source"],
                    "score": r["score"],
                    "metadata": {},
                }
                for r in results
            ]

        else:
            raise ValueError(f"Unknown RAG type: {self.rag_type}")
