from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class RaptorNode:
    id: str
    text: str
    level: int
    source_doc: str
    embedding: Optional[list] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class BuildProgress:
    status: str
    current_level: int
    total_levels: int
    clusters_completed: int
    total_clusters_current_level: int
    nodes_created: int
    last_updated: datetime
