import json
from dataclasses import asdict
from typing import Any, Dict, Optional

from .common import _deep_merge, _extract_top_level_extra, _get_section, _with_extra
from .data import DataConfig
from .feature import FeatureConfig
from .preprocess import PreprocessConfig
from .training import TrainingConfig


class PipelineConfig:
    """Root pipeline config with nested sections and JSON IO."""

    def __init__(
        self,
        json_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ):
        payload: Dict[str, Any] = {}
        if json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        if config_dict:
            payload = _deep_merge(payload, config_dict)
        if overrides:
            payload = _deep_merge(payload, overrides)

        self.data = DataConfig.from_dict(_get_section(payload, "DataConfig", "data", "data_config"))
        self.preprocess = PreprocessConfig.from_dict(
            _get_section(payload, "PreprocessConfig", "preprocess", "preprocess_config")
        )
        self.feature = FeatureConfig.from_dict(_get_section(payload, "FeatureConfig", "feature", "feature_config"))
        self.training = TrainingConfig.from_dict(_get_section(payload, "TrainingConfig", "training", "training_config"))

        self.extra = _extract_top_level_extra(payload)

    @classmethod
    def from_json(cls, json_path: str) -> "PipelineConfig":
        return cls(json_path=json_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "DataConfig": _with_extra(asdict(self.data), self.data.extra),
            "PreprocessConfig": _with_extra(asdict(self.preprocess), self.preprocess.extra),
            "FeatureConfig": _with_extra(asdict(self.feature), self.feature.extra),
            "TrainingConfig": _with_extra(asdict(self.training), self.training.extra),
            **self.extra,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=True)
