import os
import pickle
import json
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RaptorPersistence:
    """
    Maneja persistencia de RAPTOR:
    - checkpoints
    - progreso
    - integración local / GCS
    """

    def __init__(
        self,
        collection_name: str,
        base_dir: str = "./.raptor_checkpoints",
    ):
        self.collection_name = collection_name
        self.base_dir = base_dir
        self.collection_dir = os.path.join(base_dir, collection_name)

        self.chroma_dir = os.path.join(self.collection_dir, "chroma")
        self.checkpoint_path = os.path.join(
            self.collection_dir, "raptor_checkpoint.pkl"
        )
        self.progress_path = os.path.join(self.collection_dir, "build_progress.json")

        os.makedirs(self.collection_dir, exist_ok=True)

    # =====================================================
    # CHECKPOINTS
    # =====================================================

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self.checkpoint_path)

    def save_checkpoint(
        self,
        nodes: Dict,
        tree_structure: Dict,
        extra: Optional[Dict] = None,
    ) -> None:
        data = {
            "nodes": nodes,
            "tree_structure": tree_structure,
            "timestamp": datetime.now().isoformat(),
        }

        if extra:
            data.update(extra)

        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"💾 Checkpoint saved: {self.checkpoint_path}")

    def load_checkpoint(self) -> Optional[Dict]:
        if not self.checkpoint_exists():
            logger.warning("⚠️ No checkpoint found")
            return None

        with open(self.checkpoint_path, "rb") as f:
            data = pickle.load(f)

        logger.info("✅ Checkpoint loaded")
        return data

    # =====================================================
    # PROGRESS
    # =====================================================

    def save_progress(self, progress: Dict) -> None:
        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def load_progress(self) -> Optional[Dict]:
        if not os.path.exists(self.progress_path):
            return None

        with open(self.progress_path, "r") as f:
            return json.load(f)

    # =====================================================
    # LEVEL CHECKPOINTS (OPCIONAL)
    # =====================================================

    def level_checkpoint_path(self, level: int) -> str:
        level_dir = os.path.join(self.collection_dir, "level_checkpoints")
        os.makedirs(level_dir, exist_ok=True)
        return os.path.join(level_dir, f"level_{level}.pkl")

    def save_level_checkpoint(
        self,
        level: int,
        nodes: Dict,
        tree_structure: Dict,
    ) -> None:
        path = self.level_checkpoint_path(level)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "level": level,
                    "nodes": nodes,
                    "tree_structure": tree_structure,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )

        logger.info(f"💾 Level {level} checkpoint saved")

    def load_level_checkpoint(self, level: int) -> Optional[Dict]:
        path = self.level_checkpoint_path(level)
        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return pickle.load(f)
