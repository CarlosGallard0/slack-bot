from typing import List, Dict, Optional, Literal
import logging

from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class RaptorRetriever:
    """
    RAPTOR Retriever
    ----------------
    Exclusive responsibility:
    - vector-based search
    - tree navigation
    - context assembly
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        nodes: Dict,
        tree_structure: Dict,
        default_mode: Literal["collapsed", "tree_traversal"] = "collapsed",
    ):
        self.vectorstore = vectorstore
        self.nodes = nodes
        self.tree_structure = tree_structure
        self.default_mode = default_mode

    # =====================================================
    # PUBLIC API
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[Literal["collapsed", "tree_traversal"]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Unified retrieval entry point.
        """
        mode = mode or self.default_mode

        if mode == "collapsed":
            return self.retrieve_collapsed(query, top_k, **kwargs)
        elif mode == "tree_traversal":
            return self.retrieve_tree_traversal(query, top_k, **kwargs)
        else:
            raise ValueError(f"Invalid retrieval mode: {mode}")

    # =====================================================
    # COLLAPSED RETRIEVAL
    # =====================================================

    def retrieve_collapsed(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Flat similarity search over the entire RAPTOR tree.
        """

        filter_dict = {}
        if source_filter:
            filter_dict["source_doc"] = source_filter

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k * 3,
            filter=filter_dict if filter_dict else None,
        )

        formatted = []
        for doc, score in results[:top_k]:
            formatted.append(
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "level": doc.metadata.get("level"),
                    "node_id": doc.metadata.get("node_id"),
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "source_doc": doc.metadata.get("source_doc", ""),
                    "metadata": doc.metadata,
                }
            )

        return formatted

    # =====================================================
    # TREE TRAVERSAL RETRIEVAL
    # =====================================================

    def retrieve_tree_traversal(
        self,
        query: str,
        top_k: int = 2,
        start_level: Optional[int] = None,
    ) -> List[Dict]:
        """
        Hierarchical top-down search across tree levels
        """

        if not self.tree_structure:
            logger.warning("⚠️ Tree not loaded")
            return []

        max_level = max(self.tree_structure.keys())
        current_level = start_level or max_level
        parent_ids = None

        selected: List[Dict] = []

        while current_level >= 0:
            level_filter = {"level": current_level}

            if parent_ids is None:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=top_k * 3,
                    filter=level_filter,
                )
            else:
                candidate_ids = []
                for pid in parent_ids:
                    node = self.nodes.get(pid)
                    if node:
                        candidate_ids.extend(node.children_ids)

                if not candidate_ids:
                    break

                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=len(candidate_ids) * 2,
                    filter=level_filter,
                )

                results = [
                    (doc, score)
                    for doc, score in results
                    if doc.metadata.get("node_id") in candidate_ids
                ]

            if not results:
                break

            level_selected = results[:top_k]

            for doc, score in level_selected:
                selected.append(
                    {
                        "text": doc.page_content,
                        "score": float(score),
                        "level": doc.metadata.get("level"),
                        "node_id": doc.metadata.get("node_id"),
                        "parent_id": doc.metadata.get("parent_id", ""),
                        "source_doc": doc.metadata.get("source_doc", ""),
                        "metadata": doc.metadata,
                    }
                )

            parent_ids = [doc.metadata.get("node_id") for doc, _ in level_selected]
            current_level -= 1

        return selected

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        include_parents: bool = False,
    ) -> str:
        """
        Returns formatted context ready for LLMs or Slackbot consumption.
        """

        results = self.retrieve(query, top_k, mode=mode)

        context_parts = []
        seen = set()

        for r in results:
            node_id = r["node_id"]
            if node_id in seen:
                continue

            header = f"[Level {r['level']}] [Source: {r['source_doc']}]"
            context_parts.append(f"{header}\n{r['text']}")
            seen.add(node_id)

            if include_parents and r.get("parent_id"):
                pid = r["parent_id"]
                parent = self.nodes.get(pid)
                if parent and pid not in seen:
                    parent_header = (
                        f"[Level {parent.level} - Summary] "
                        f"[Source: {parent.source_doc}]"
                    )
                    context_parts.append(f"{parent_header}\n{parent.text}")
                    seen.add(pid)

        return "\n\n---\n\n".join(context_parts)

    # =====================================================
    # STATS
    # =====================================================

    def get_stats(self) -> Dict:
        """
        Basic statistics about the loaded RAPTOR tree.
        """
        return {
            "total_nodes": len(self.nodes),
            "levels": len(self.tree_structure),
            "nodes_per_level": {
                level: len(nodes) for level, nodes in self.tree_structure.items()
            },
            "sources": list({node.source_doc for node in self.nodes.values()}),
            "default_mode": self.default_mode,
        }
