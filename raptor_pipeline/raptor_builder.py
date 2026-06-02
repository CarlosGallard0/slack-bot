import os
import asyncio
import logging
from typing import List, Dict, Optional
import numpy as np

from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma.vectorstores import Chroma

from sklearn.mixture import GaussianMixture
import umap

from raptor_pipeline.raptor_models import RaptorNode
from raptor_pipeline.raptor_persistence import RaptorPersistence
from raptor_pipeline.raptor_utils import (
    generate_node_id,
    truncate_text,
    safe_async_run,
)
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class RaptorBuilder:
    """
    RAPTOR Builder
    --------------
    Responsable EXCLUSIVO de:
    - construir el árbol RAPTOR
    - embeddings
    - clustering
    - resúmenes
    - delegar persistencia
    """

    # =====================================================
    # INIT
    # =====================================================

    def __init__(
        self,
        collection_name: str,
        max_depth: int = 3,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        persist_directory: Optional[str] = None,
        cluster_batch_size: int = 5,
        embeddings: Optional[Embeddings] = None,
        llm: Optional[BaseLanguageModel] = None,
    ):
        self.collection_name = collection_name
        self.max_depth = max_depth
        self.cluster_batch_size = cluster_batch_size

        # Persistence manager
        self.persistence = RaptorPersistence(
            collection_name=collection_name,
            base_dir=persist_directory or "./.raptor_checkpoints",
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Embeddings
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model="text-embedding-005",
            vertexai=True,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # LLM
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            vertexai=True,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persistence.chroma_dir,
        )

        # Tree state
        self.nodes: Dict[str, RaptorNode] = {}
        self.tree_structure: Dict[int, List[str]] = {}

        # Prompt
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You summarize text accurately and concisely."),
                ("user", "{text}"),
            ]
        )

    # =====================================================
    # PUBLIC API
    # =====================================================

    def build_tree(
        self,
        documents: List[Document],
        force_rebuild: bool = False,
    ) -> None:
        logger.info("🌳 Starting RAPTOR build")

        if not force_rebuild and self.persistence.checkpoint_exists():
            logger.info("🔄 Existing checkpoint found, loading")
            data = self.persistence.load_checkpoint()
            if data:
                self.nodes = data["nodes"]
                self.tree_structure = data["tree_structure"]
                return

        self._build_from_scratch(documents)
        self._add_to_vectorstore()

        self.persistence.save_checkpoint(
            nodes=self.nodes,
            tree_structure=self.tree_structure,
        )

    @classmethod
    def from_existing(
        cls,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> "RaptorBuilder":
        instance = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        data = instance.persistence.load_checkpoint()
        if data:
            instance.nodes = data["nodes"]
            instance.tree_structure = data["tree_structure"]
        return instance

    # =====================================================
    # BUILD LOGIC
    # =====================================================

    def _build_from_scratch(self, documents: List[Document]) -> None:
        self.nodes.clear()
        self.tree_structure.clear()

        # -------- Level 0 --------
        chunks = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            for text in self.text_splitter.split_text(doc.page_content):
                chunks.append({"text": text, "source": source})

        logger.info(f"📄 Level 0 chunks: {len(chunks)}")
        level_nodes = self._create_leaf_nodes(chunks)
        self.tree_structure[0] = [n.id for n in level_nodes]

        # -------- Upper levels --------
        for level in range(1, self.max_depth + 1):
            if len(level_nodes) <= 1:
                break

            logger.info(f"📊 Building level {level}")
            level_nodes = self._build_level(level_nodes, level)
            self.tree_structure[level] = [n.id for n in level_nodes]

    def _build_level(
        self,
        child_nodes: List[RaptorNode],
        level: int,
    ) -> List[RaptorNode]:

        texts = [n.text for n in child_nodes]
        embeddings = safe_async_run(self.embeddings.aembed_documents(texts))
        emb_array = np.array(embeddings)

        for node, emb in zip(child_nodes, emb_array):
            node.embedding = emb

        clusters = self._cluster_nodes(child_nodes, emb_array)
        parent_nodes = safe_async_run(
            self._process_clusters(child_nodes, clusters, level)
        )
        return parent_nodes

    # =====================================================
    # CLUSTER PROCESSING
    # =====================================================
    async def _process_single_cluster(
        self,
        nodes: List[RaptorNode],
        cluster_ids: List[str],
        level: int,
        index: int,
    ) -> RaptorNode:

        cluster_nodes = [n for n in nodes if n.id in cluster_ids]
        combined_text = "\n\n".join(n.text for n in cluster_nodes)

        summary = await self._summarize(combined_text)

        parent_id = generate_node_id(level, index)
        parent = RaptorNode(
            id=parent_id,
            text=summary,
            level=level,
            source_doc=cluster_nodes[0].source_doc,
            children_ids=[n.id for n in cluster_nodes],
            metadata={"is_summary": True},
        )

        for child in cluster_nodes:
            child.parent_id = parent_id

        self.nodes[parent.id] = parent
        return parent

    async def _process_clusters(
        self,
        nodes: List[RaptorNode],
        clusters: List[List[str]],
        level: int,
    ) -> List[RaptorNode]:

        parents: List[RaptorNode] = []

        for i in range(0, len(clusters), self.cluster_batch_size):
            batch = clusters[i : i + self.cluster_batch_size]

            tasks = [
                self._process_single_cluster(
                    nodes,
                    cluster_ids,
                    level,
                    i + idx,
                )
                for idx, cluster_ids in enumerate(batch)
            ]

            results = await asyncio.gather(*tasks)
            parents.extend(results)

        return parents

    # =====================================================
    # HELPERS
    # =====================================================

    def _create_leaf_nodes(self, chunks: List[Dict]) -> List[RaptorNode]:
        nodes = []
        for idx, ch in enumerate(chunks):
            node = RaptorNode(
                id=generate_node_id(0, idx),
                text=ch["text"],
                level=0,
                source_doc=ch["source"],
            )
            self.nodes[node.id] = node
            nodes.append(node)
        return nodes

    def _cluster_nodes(
        self,
        nodes: List[RaptorNode],
        embeddings: np.ndarray,
    ) -> List[List[str]]:

        if len(nodes) <= 3:
            return [[n.id for n in nodes]]

        reducer = umap.UMAP(
            n_neighbors=min(15, len(nodes) - 1),
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)

        n_clusters = min(max(2, len(nodes) // 5), 10)
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            covariance_type="diag",
        )
        labels = gmm.fit_predict(reduced)

        clusters: Dict[int, List[str]] = {}
        for node, label in zip(nodes, labels):
            clusters.setdefault(label, []).append(node.id)

        return list(clusters.values())

    async def _summarize(self, text: str) -> str:
        safe_text = truncate_text(text)
        chain = self.summary_prompt | self.llm
        response = await chain.ainvoke({"text": safe_text})
        return response.content

    # =====================================================
    # VECTORSTORE
    # =====================================================

    def _add_to_vectorstore(self) -> None:
        ids, texts, metas = [], [], []

        for node in self.nodes.values():
            ids.append(node.id)
            texts.append(node.text)
            metas.append(
                {
                    "node_id": node.id,
                    "level": node.level,
                    "parent_id": node.parent_id or "",
                    "children_ids": ",".join(node.children_ids),
                    "source_doc": node.source_doc,
                    **node.metadata,
                }
            )

        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metas,
            ids=ids,
        )
        logger.info(f"✅ {len(ids)} nodes added to vectorstore")
