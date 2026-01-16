import pickle
import os
from typing import List, Dict, Optional, Literal, Union
from dataclasses import dataclass, field, asdict
import numpy as np
from datetime import datetime
import logging
from enum import Enum
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
import json
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BuildStatus(Enum):
    """Estados posibles de construcción del árbol"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    LEVEL_COMPLETED = "level_completed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BuildProgress:
    """Registro detallado del progreso de construcción"""

    status: str
    current_level: int
    total_levels: int
    clusters_completed: int
    total_clusters_current_level: int
    nodes_created: int
    last_updated: datetime
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict:
        return {**asdict(self), "last_updated": self.last_updated.isoformat()}

    @classmethod
    def from_dict(cls, data: Dict) -> "BuildProgress":
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


@dataclass
class RaptorNode:
    """Nodo en el árbol RAPTOR"""

    id: str
    text: str
    level: int
    source_doc: str
    embedding: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class RaptorLangChain:
    def __init__(
        self,
        embeddings: Optional[Union[Embeddings, str]] = None,
        llm: Optional[Union[BaseLanguageModel, str]] = None,
        vectorstore: Optional[VectorStore] = None,
        collection_name: str = "raptor_tree",
        max_depth: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_mode: Literal["collapsed", "tree_traversal"] = "collapsed",
        persist_directory: Optional[str] = None,
        checkpoint_frequency: Literal["cluster", "level", "final"] = "level",
        enable_detailed_logging: bool = True,
        cluster_batch_size: int = 5,
        google_api_key: Optional[str] = None,
        use_vertexai: bool = True, 
        embedding_model: Optional[str] = None, 
        llm_model: Optional[str] = None, 
    ):
        self.collection_name = collection_name
        self.max_depth = max_depth
        self.retrieval_mode = retrieval_mode
        self.checkpoint_frequency = checkpoint_frequency
        self.enable_detailed_logging = enable_detailed_logging
        self.cluster_batch_size = cluster_batch_size

        # Configuración de Google
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.use_vertexai = use_vertexai

        # ============================================
        # DIRECTORIOS
        # ============================================
        if persist_directory is None:
            persist_directory = f"./.raptor_checkpoints/{collection_name}"
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.logs_directory = os.path.join(self.persist_directory, "logs")
        os.makedirs(self.logs_directory, exist_ok=True)

        # Logger
        if enable_detailed_logging:
            self.logger = self._setup_detailed_logger()
        else:
            self.logger = logger

        self.logger.info(f"🔧 Cluster batch size: {self.cluster_batch_size}")

        # ============================================
        # EMBEDDINGS: Google Generative AI
        # ============================================
        if embeddings is None:
            self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
                model="text-embedding-005",
                vertexai=True,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            print("✅ Embeddings: text-embedding-005")
        else:
            self.embeddings = embeddings
            print(f"✅ Embeddings custom: {type(embeddings).__name__}")

        # ============================================
        # LLM: Google Gemini
        # ============================================
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=1.0,
                vertexai=True,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )

            print("✅ LLM: gemini-2.5-flash")
        else:
            self.llm = llm
            print(f"✅ LLM custom: {type(llm).__name__}")

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # ============================================
        # VECTORSTORE: Chroma (independiente de proveedor)
        # ============================================
        if vectorstore is None:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=os.path.join(self.persist_directory, "chroma"),
            )
            self.logger.info("🔧 Vector store: Chroma")
        else:
            self.vectorstore = vectorstore
            self.logger.info(
                f"✅ Vector store personalizado: {type(vectorstore).__name__}"
            )

        self.nodes: Dict[str, RaptorNode] = {}
        self.tree_structure: Dict[int, List[str]] = {}
        self.build_progress: Optional[BuildProgress] = None

        # Prompt para resúmenes
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that creates concise summaries.",
                ),
                (
                    "user",
                    "Summarize the following text, including as many key details as needed:\n\n{text}",
                ),
            ]
        )

    def _setup_detailed_logger(self) -> logging.Logger:
        """Configura un logger detallado con archivo"""
        detailed_logger = logging.getLogger(f"raptor.{self.collection_name}")
        detailed_logger.setLevel(logging.DEBUG)

        log_file = os.path.join(
            self.logs_directory, f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        detailed_logger.addHandler(file_handler)
        detailed_logger.addHandler(console_handler)

        # FIX: Usar detailed_logger, no self.logger
        detailed_logger.info(f"📝 Log detallado: {log_file}")

        return detailed_logger
    
    def retrieve_collapsed(
        self, query: str, top_k: int = 5, source_filter: Optional[str] = None
    ) -> List[Dict]:
        filter_dict = {}
        if source_filter:
            filter_dict["source_doc"] = source_filter

        results = self.vectorstore.similarity_search_with_score(
            query=query, k=top_k * 3, filter=filter_dict if filter_dict else None
        )

        formatted_results = []
        for doc, score in results[:top_k]:
            formatted_results.append(
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "level": doc.metadata["level"],
                    "node_id": doc.metadata["node_id"],
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "source_doc": doc.metadata.get("source_doc", ""),
                    "metadata": doc.metadata,
                }
            )

        return formatted_results

    def retrieve_tree_traversal(
        self, query: str, top_k: int = 2, start_level: Optional[int] = None
    ) -> List[Dict]:
        if not self.tree_structure:
            self.logger.warning("⚠️ Árbol no construido")
            return []

        max_level = max(self.tree_structure.keys())
        if start_level is None:
            start_level = max_level

        selected_nodes = []
        current_level = start_level
        parent_ids = None

        while current_level >= 0:
            filter_dict = {"level": current_level}

            if parent_ids is None:
                level_node_count = len(self.tree_structure.get(current_level, []))
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=min(top_k * 3, level_node_count),
                    filter=filter_dict,
                )
            else:
                candidate_ids = []
                for parent_id in parent_ids:
                    parent_node = self.nodes.get(parent_id)
                    if parent_node:
                        candidate_ids.extend(parent_node.children_ids)

                if not candidate_ids:
                    break

                results = self.vectorstore.similarity_search_with_score(
                    query=query, k=len(candidate_ids) * 2, filter=filter_dict
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
                selected_nodes.append(
                    {
                        "text": doc.page_content,
                        "score": float(score),
                        "level": doc.metadata["level"],
                        "node_id": doc.metadata["node_id"],
                        "parent_id": doc.metadata.get("parent_id", ""),
                        "source_doc": doc.metadata.get("source_doc", ""),
                        "metadata": doc.metadata,
                    }
                )

            parent_ids = [doc.metadata["node_id"] for doc, _ in level_selected]
            current_level -= 1

        return selected_nodes

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[Literal["collapsed", "tree_traversal"]] = None,
        **kwargs,
    ) -> List[Dict]:
        mode = mode or self.retrieval_mode

        if mode == "collapsed":
            return self.retrieve_collapsed(query, top_k, **kwargs)
        elif mode == "tree_traversal":
            return self.retrieve_tree_traversal(query, top_k, **kwargs)
        else:
            raise ValueError(f"Modo inválido: {mode}")

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        include_parents: bool = False,
    ) -> str:
        results = self.retrieve(query, top_k, mode=mode)

        context_parts = []
        seen_ids = set()

        for result in results:
            node_id = result["node_id"]

            if node_id in seen_ids:
                continue

            header = f"[Level {result['level']}] [Source: {result['source_doc']}]"
            context_parts.append(f"{header}\n{result['text']}")
            seen_ids.add(node_id)

            if include_parents and result["parent_id"]:
                parent_id = result["parent_id"]
                if parent_id in self.nodes and parent_id not in seen_ids:
                    parent = self.nodes[parent_id]
                    parent_header = f"[Level {parent.level} - Summary] [Source: {parent.source_doc}]"
                    context_parts.append(f"{parent_header}\n{parent.text}")
                    seen_ids.add(parent_id)

        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def collection_exists(
        collection_name: str, persist_directory: Optional[str] = None
    ) -> bool:
        if persist_directory is None:
            persist_directory = f"./.raptor_checkpoints/{collection_name}"

        checkpoint_path = os.path.join(persist_directory, "raptor_checkpoint.pkl")
        chroma_path = os.path.join(persist_directory, "chroma")

        if not (os.path.exists(checkpoint_path) and os.path.exists(chroma_path)):
            return False

        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
                return len(data.get("nodes", {})) > 0
        except Exception:
            return False

    def _get_checkpoint_path(
        self, level: Optional[int] = None, cluster: Optional[int] = None
    ) -> str:
        if level is None:
            return os.path.join(self.persist_directory, "raptor_checkpoint.pkl")

        checkpoint_dir = os.path.join(self.persist_directory, "level_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if cluster is None:
            return os.path.join(checkpoint_dir, f"level_{level}.pkl")
        else:
            cluster_dir = os.path.join(checkpoint_dir, f"level_{level}_clusters")
            os.makedirs(cluster_dir, exist_ok=True)
            return os.path.join(cluster_dir, f"cluster_{cluster}.pkl")

    def _get_progress_path(self) -> str:
        return os.path.join(self.persist_directory, "build_progress.json")

    def _save_progress(self) -> None:
        if self.build_progress is None:
            return

        with open(self._get_progress_path(), "w") as f:
            json.dump(self.build_progress.to_dict(), f, indent=2)

    def _load_progress(self) -> Optional[BuildProgress]:
        progress_path = self._get_progress_path()
        if not os.path.exists(progress_path):
            return None

        try:
            with open(progress_path, "r") as f:
                data = json.load(f)
            return BuildProgress.from_dict(data)
        except Exception as e:
            self.logger.error(f"❌ Error cargando progreso: {e}")
            return None

    def checkpoint_exists(self) -> bool:
        checkpoint_path = self._get_checkpoint_path()
        chroma_path = os.path.join(self.persist_directory, "chroma")

        if not (os.path.exists(checkpoint_path) and os.path.exists(chroma_path)):
            return False

        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
                return len(data.get("nodes", {})) > 0
        except Exception as e:
            return False

    def _save_level_checkpoint(self, level: int) -> None:
        checkpoint_data = {
            "nodes": self.nodes,
            "tree_structure": self.tree_structure,
            "level": level,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self._get_checkpoint_path(level=level), "wb") as f:
            pickle.dump(checkpoint_data, f)

        self.logger.info(
            f"💾 Checkpoint nivel {level} guardado: {len(self.nodes)} nodos"
        )

    def _save_cluster_checkpoint(
        self, level: int, cluster_idx: int, parent_node: RaptorNode
    ) -> None:
        checkpoint_data = {
            "parent_node": parent_node,
            "timestamp": datetime.now().isoformat(),
        }

        with open(
            self._get_checkpoint_path(level=level, cluster=cluster_idx), "wb"
        ) as f:
            pickle.dump(checkpoint_data, f)

    def save_checkpoint(self) -> None:
        if not self.nodes:
            self.logger.warning("⚠️  No hay árbol para guardar")
            return

        checkpoint_data = {
            "nodes": self.nodes,
            "tree_structure": self.tree_structure,
            "max_depth": self.max_depth,
            "collection_name": self.collection_name,
            "retrieval_mode": self.retrieval_mode,
            "build_progress": (
                self.build_progress.to_dict() if self.build_progress else None
            ),
            "timestamp": datetime.now().isoformat(),
        }

        with open(self._get_checkpoint_path(), "wb") as f:
            pickle.dump(checkpoint_data, f)

        self.logger.info("💾 Checkpoint guardado")

    def load_checkpoint(self) -> bool:
        if not self.checkpoint_exists():
            self.logger.info("ℹ️  No checkpoint existente")
            return False

        try:
            with open(self._get_checkpoint_path(), "rb") as f:
                checkpoint_data = pickle.load(f)

            self.nodes = checkpoint_data["nodes"]
            self.tree_structure = checkpoint_data["tree_structure"]
            self.max_depth = checkpoint_data.get("max_depth", self.max_depth)
            self.retrieval_mode = checkpoint_data.get(
                "retrieval_mode", self.retrieval_mode
            )

            if (
                "build_progress" in checkpoint_data
                and checkpoint_data["build_progress"]
            ):
                self.build_progress = BuildProgress.from_dict(
                    checkpoint_data["build_progress"]
                )

            self.logger.info(f"✅ Checkpoint cargado: {len(self.nodes)} nodos")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error al cargar checkpoint: {e}")
            return False

    def _load_level_checkpoint(self, level: int) -> bool:
        checkpoint_path = self._get_checkpoint_path(level=level)
        if not os.path.exists(checkpoint_path):
            return False

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)

            self.nodes = checkpoint_data["nodes"]
            self.tree_structure = checkpoint_data["tree_structure"]
            return True
        except Exception as e:
            return False


    @classmethod
    def from_existing(
        cls,
        collection_name: str,
        persist_directory: Optional[str] = None,
        embeddings: Optional[Union[Embeddings, str]] = None,
        llm: Optional[Union[BaseLanguageModel, str]] = None,
        google_api_key: Optional[str] = None,
    ) -> Optional["RaptorLangChain"]:
        if not cls.collection_exists(collection_name, persist_directory):
            return None

        instance = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embeddings=embeddings,
            llm=llm,
            google_api_key=google_api_key,
        )

        if instance.load_checkpoint():
            return instance

        return None