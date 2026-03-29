import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_bool_or_int(value: Any) -> Union[bool, int]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"false", "none", "off"}:
            return False
        if v.lower() in {"true", "on"}:
            return True
        if v.isdigit():
            return int(v)
    raise ValueError(f"Expected bool|int compatible value, got: {value!r}")


def _split_known(d: Dict[str, Any], known_keys: set) -> tuple[Dict[str, Any], Dict[str, Any]]:
    known: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in d.items():
        if k in known_keys:
            known[k] = v
        else:
            extra[k] = v
    return known, extra


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


@dataclass
class PreprocessConfig:
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
            "only_char",
            "lower",
            "stop_remove",
            "hfw_proportion",
            "freq_proportion",
            "min_freq_threshold",
        }
        known, extra = _split_known(data, known_keys)
        for bool_key in ("only_char", "lower", "stop_remove"):
            if bool_key in known:
                known[bool_key] = _coerce_bool(known[bool_key])
        cfg = cls(**known)
        cfg.extra = extra
        return cfg


@dataclass
class FeatureConfig:
    max_vocab: Union[bool, int] = False
    encoder_type: Union[bool, str] = "w2v"
    use_hand_features: bool = True
    allow_recommendation_fallback: bool = True
    recommendation_threshold: float = 3.5
    drop_post_review_leakage: bool = True
    preserve_corpus_cache: bool = True
    shuffle_seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_vocab_token(self) -> str:
        return "False" if self.max_vocab is False else str(self.max_vocab)

    @property
    def encoder_token(self) -> str:
        return "False" if self.encoder_type is False else str(self.encoder_type)

    @property
    def hand_token(self) -> str:
        return "True" if self.use_hand_features else "False"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FeatureConfig":
        data = data or {}
        known_keys = {
            "max_vocab",
            "encoder_type",
            "use_hand_features",
            "allow_recommendation_fallback",
            "recommendation_threshold",
            "drop_post_review_leakage",
            "preserve_corpus_cache",
            "shuffle_seed",
        }
        known, extra = _split_known(data, known_keys)

        if "max_vocab" in known:
            known["max_vocab"] = _coerce_bool_or_int(known["max_vocab"])
        for bool_key in (
            "use_hand_features",
            "allow_recommendation_fallback",
            "drop_post_review_leakage",
            "preserve_corpus_cache",
        ):
            if bool_key in known:
                known[bool_key] = _coerce_bool(known[bool_key])

        cfg = cls(**known)
        cfg.extra = extra
        return cfg


def _default_model_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "name": "LogReg L2 lambda=0.01",
            "model_type": "LogisticRegression",
            "requires_scaling": True,
            "hyperparams": {"balance": True, "penalty": "l2", "reg_lambda": 0.01, "lr": 0.05, "max_iter": 2500, "random_state": 42},
        },
        {
            "name": "LogReg L2 lambda=0.001",
            "model_type": "LogisticRegression",
            "requires_scaling": True,
            "hyperparams": {"balance": True, "penalty": "l2", "reg_lambda": 0.001, "lr": 0.04, "max_iter": 2500, "random_state": 42},
        },
        {
            "name": "SVM Linear C=1",
            "model_type": "SVMClassifier",
            "requires_scaling": True,
            "hyperparams": {"balance": True, "kernel": "linear", "C": 1.0, "max_iter_linear": 6000, "random_state": 42},
        },
        {
            "name": "SVM Linear C=2",
            "model_type": "SVMClassifier",
            "requires_scaling": True,
            "hyperparams": {"balance": True, "kernel": "linear", "C": 2.0, "max_iter_linear": 7000, "random_state": 42},
        },
        {
            "name": "RF n=300 depth=10",
            "model_type": "RandomForestClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2, "random_state": 42},
        },
        {
            "name": "RF n=500 depth=12",
            "model_type": "RandomForestClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 500, "max_depth": 12, "min_samples_leaf": 2, "random_state": 42},
        },
        {
            "name": "Ada depth=1 n=400 lr=0.1",
            "model_type": "AdaBoostClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 400, "learning_rate": 0.1, "base_max_depth": 1, "random_state": 42},
        },
        {
            "name": "Ada depth=2 n=400 lr=0.15",
            "model_type": "AdaBoostClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 400, "learning_rate": 0.15, "base_max_depth": 2, "random_state": 42},
        },
        {
            "name": "Ada depth=3 n=300 lr=0.1",
            "model_type": "AdaBoostClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 300, "learning_rate": 0.1, "base_max_depth": 3, "random_state": 42},
        },
        {
            "name": "GB depth=2 n=500 lr=0.05",
            "model_type": "GradientBoostingClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 500, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 3, "random_state": 42},
        },
        {
            "name": "GB depth=3 n=500 lr=0.05",
            "model_type": "GradientBoostingClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 3, "random_state": 42},
        },
        {
            "name": "GB depth=3 n=700 lr=0.03",
            "model_type": "GradientBoostingClassifier",
            "requires_scaling": False,
            "hyperparams": {"balance": True, "n_estimators": 700, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 3, "random_state": 42},
        },
    ]


@dataclass
class TrainingConfig:
    enabled: bool = True
    data_dir: Optional[str] = None
    standardize_linear_models: bool = True
    threshold_min: float = 0.2
    threshold_max: float = 0.8
    threshold_steps: int = 25
    use_probability_midpoints: bool = True
    ensemble_top_k: int = 3
    model_candidates: List[Dict[str, Any]] = field(default_factory=_default_model_candidates)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        data = data or {}
        known_keys = {
            "enabled",
            "data_dir",
            "standardize_linear_models",
            "threshold_min",
            "threshold_max",
            "threshold_steps",
            "use_probability_midpoints",
            "ensemble_top_k",
            "model_candidates",
        }
        known, extra = _split_known(data, known_keys)
        for bool_key in ("enabled", "standardize_linear_models", "use_probability_midpoints"):
            if bool_key in known:
                known[bool_key] = _coerce_bool(known[bool_key])
        cfg = cls(**known)
        cfg.extra = extra
        return cfg


class Config:
    """
    Root pipeline config with nested sections and JSON IO.

    Expected JSON sections:
    - DataConfig
    - PreprocessConfig
    - FeatureConfig
    - TrainingConfig
    """

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
    def from_json(cls, json_path: str) -> "Config":
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


def _get_section(payload: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for k in keys:
        value = payload.get(k)
        if isinstance(value, dict):
            return value
    return {}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in update.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _with_extra(base_dict: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base_dict)
    out.pop("extra", None)
    out.update(extra)
    return out


def _extract_top_level_extra(payload: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {
        "DataConfig",
        "data",
        "data_config",
        "PreprocessConfig",
        "preprocess",
        "preprocess_config",
        "FeatureConfig",
        "feature",
        "feature_config",
        "TrainingConfig",
        "training",
        "training_config",
    }
    return {k: v for k, v in payload.items() if k not in reserved}
