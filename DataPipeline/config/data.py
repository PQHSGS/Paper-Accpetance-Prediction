import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import _split_known


@dataclass
class DataConfig:
    base_dir: str = "Dataset"
    combined_name: str = "all_combined"
    datasets: List[str] = field(default_factory=lambda: [
        "acl_2017",
        "iclr_2017",
        "arxiv.cs.ai_2007-2017",
        "arxiv.cs.cl_2007-2017",
        "arxiv.cs.lg_2007-2017",
        "conll_2016",
    ])
    splits: List[str] = field(default_factory=lambda: ["train", "dev", "test"])
    reviews_subdir: str = "reviews"
    parsed_subdir: str = "parsed_pdfs"
    output_subdir: str = "dataset"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def combined_dir(self) -> str:
        return os.path.join(self.base_dir, self.combined_name)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DataConfig":
        data = data or {}
        known_keys = {
            "base_dir",
            "combined_name",
            "datasets",
            "splits",
            "reviews_subdir",
            "parsed_subdir",
            "output_subdir",
        }
        known, extra = _split_known(data, known_keys)
        cfg = cls(**known)
        cfg.extra = extra
        return cfg
