from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import _coerce_bool, _coerce_str_list, _split_known


@dataclass
class PreprocessConfig:
    methods: List[str] = field(default_factory=lambda: [
        "build_corpus_words",
        "compute_frequency_buckets",
    ])
    only_char: bool = True
    lower: bool = True
    stop_remove: bool = True
    hfw_proportion: float = 0.01
    freq_proportion: float = 0.05
    min_freq_threshold: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PreprocessConfig":
        data = data or {}
        known_keys = {
            "methods",
            "only_char",
            "lower",
            "stop_remove",
            "hfw_proportion",
            "freq_proportion",
            "min_freq_threshold",
        }
        known, extra = _split_known(data, known_keys)
        if "methods" in known:
            known["methods"] = _coerce_str_list(known["methods"], "PreprocessConfig.methods")
        for bool_key in ("only_char", "lower", "stop_remove"):
            if bool_key in known:
                known[bool_key] = _coerce_bool(known[bool_key])
        cfg = cls(**known)
        cfg.extra = extra
        return cfg
