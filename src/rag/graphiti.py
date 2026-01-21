import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GraphitiClient:
    """Manages Graphiti knowledge graph operations."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """
        Initialize Graphiti client.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")

        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_choice = os.getenv("MODEL_NAME", "gpt-4.1-mini")

        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
            
        self.embedding_base_url = os.getenv(
            "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
        )
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(768)

        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable not set")

        self.graphiti: Optional[Graphiti] = None
        self._initialized = False

    async def initialize(self):
        """Initialize Graphiti client."""
        try:
            current_loop = asyncio.get_running_loop()
            if self._initialized and self._loop is not None and current_loop is not self._loop:
                logger.warning("Event loop changed, re-initializing Graphiti client")
                self.graphiti = None
                self._initialized = False
        except RuntimeError:
            pass

        if self._initialized:
            return

        try:
            self._loop = asyncio.get_running_loop()

            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,  
                base_url=self.llm_base_url,
            )

            llm_client = GeminiClient(config=llm_config)

            embedder = GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url,
                )
            )

            reranker = GeminiRerankerClient(client=llm_client, config=llm_config)

            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=reranker
            )

            await self.graphiti.build_indices_and_constraints()

            self._initialized = True
            logger.info(
                f"Graphiti client initialized successfully with LLM: {self.llm_choice} and embedder: {self.embedding_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise

    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")

    async def search(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.

        Args:
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search

        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            results = await self.graphiti.search(query)

            # Collect all episode UUIDs
            episode_uuids = []
            for result in results:
                if hasattr(result, "episodes") and result.episodes:
                    episode_uuids.extend(result.episodes)

            # Fetch episode details
            episode_dict = {}
            if episode_uuids:
                episodes = await EpisodicNode.get_by_uuids(self.graphiti.driver, list(set(episode_uuids)))
                episode_dict = {ep.uuid: ep.name for ep in episodes}

            return [
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": (
                        str(result.valid_at)
                        if hasattr(result, "valid_at") and result.valid_at
                        else None
                    ),
                    "invalid_at": (
                        str(result.invalid_at)
                        if hasattr(result, "invalid_at") and result.invalid_at
                        else None
                    ),
                    "source_node_uuid": (
                        str(result.source_node_uuid)
                        if hasattr(result, "source_node_uuid")
                        and result.source_node_uuid
                        else None
                    ),
                    "source_title": (
                        episode_dict.get(result.episodes[0])
                        if hasattr(result, "episodes") and result.episodes
                        else None
                    ),
                }
                for result in results
            ]

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def get_related_entities(
        self,
        entity_name: str,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Get entities related to a given entity using Graphiti search.

        Args:
            entity_name: Name of the entity
            relationship_types: Types of relationships to follow (not used with Graphiti)
            depth: Maximum depth to traverse (not used with Graphiti)

        Returns:
            Related entities and relationships
        """
        if not self._initialized:
            await self.initialize()

        results = await self.graphiti.search(f"relationships involving {entity_name}")

        related_entities = set()
        facts = []

        for result in results:
            facts.append(
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": (
                        str(result.valid_at)
                        if hasattr(result, "valid_at") and result.valid_at
                        else None
                    ),
                }
            )

            if entity_name.lower() in result.fact.lower():
                related_entities.add(entity_name)

        return {
            "central_entity": entity_name,
            "related_facts": facts,
            "search_method": "graphiti_semantic_search",
        }

    async def get_entity_timeline(
        self,
        entity_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of facts for an entity using Graphiti.

        Args:
            entity_name: Name of the entity
            start_date: Start of time range (not currently used)
            end_date: End of time range (not currently used)

        Returns:
            Timeline of facts
        """
        if not self._initialized:
            await self.initialize()

        results = await self.graphiti.search(f"timeline history of {entity_name}")

        timeline = []
        for result in results:
            timeline.append(
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": (
                        str(result.valid_at)
                        if hasattr(result, "valid_at") and result.valid_at
                        else None
                    ),
                    "invalid_at": (
                        str(result.invalid_at)
                        if hasattr(result, "invalid_at") and result.invalid_at
                        else None
                    ),
                }
            )

        timeline.sort(key=lambda x: x.get("valid_at") or "", reverse=True)

        return timeline

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.

        Returns:
            Graph statistics
        """
        if not self._initialized:
            await self.initialize()

        try:
            test_results = await self.graphiti.search("test")
            return {
                "graphiti_initialized": True,
                "sample_search_results": len(test_results),
                "note": "Detailed statistics require direct Neo4j access",
            }
        except Exception as e:
            return {"graphiti_initialized": False, "error": str(e)}

    async def clear_graph(self):
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()

        try:
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
        except Exception as e:
            logger.error(f"Failed to clear graph using clear_data: {e}")
            if self.graphiti:
                await self.graphiti.close()

            llm_config = LLMConfig(
                api_key=self.llm_api_key,
                model=self.llm_choice,
                small_model=self.llm_choice,
                base_url=self.llm_base_url,
            )

            llm_client = GeminiClient(config=llm_config)

            embedder = GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.embedding_api_key,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dimensions,
                    base_url=self.embedding_base_url,
                )
            )

            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=GeminiRerankerClient(
                    client=llm_client, config=llm_config
                ),
            )
            await self.graphiti.build_indices_and_constraints()

            logger.warning("Reinitialized Graphiti client (fresh indices created)")
