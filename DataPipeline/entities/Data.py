from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set


@dataclass
class ProcessData:
    split: str
    papers: Sequence[Any] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    hfws: List[str] = field(default_factory=list)
    most_frequent_words: Set[str] = field(default_factory=set)
    least_frequent_words: Set[str] = field(default_factory=set)
    label_source_counts: Dict[str, int] = field(default_factory=dict)
    skipped_unlabeled: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturedData:
    split: str
    rows: List[Dict[int, float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    id_to_feature: Dict[str, int] = field(default_factory=dict)
    dropped_features: Set[str] = field(default_factory=set)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
