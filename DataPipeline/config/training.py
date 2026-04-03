from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .common import _coerce_bool, _coerce_str, _coerce_str_list, _split_known


@dataclass
class TrainingConfig:
    enabled: bool = True
    data_dir: Optional[str] = None
    preprocess_methods: list[str] = field(default_factory=list)
    model: str = "LogisticRegression"
    model_param: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        data = data or {}
        known_keys = {
            "enabled",
            "data_dir",
            "preprocess_methods",
            "model",
            "model_param",
        }
        known, extra = _split_known(data, known_keys)

        if "preprocess_methods" in known:
            known["preprocess_methods"] = _coerce_str_list(
                known["preprocess_methods"],
                "TrainingConfig.preprocess_methods",
            )
        if "enabled" in known:
            known["enabled"] = _coerce_bool(known["enabled"])
        if "model" in known:
            known["model"] = _coerce_str(known["model"], "TrainingConfig.model")
        if "model_param" in known:
            if not isinstance(known["model_param"], dict):
                raise ValueError(
                    f"Expected dict for TrainingConfig.model_param, got: {known['model_param']!r}"
                )
            known["model_param"] = dict(known["model_param"])

        cfg = cls(**known)
        cfg.extra = extra
        return cfg
