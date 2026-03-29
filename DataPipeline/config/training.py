from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import _coerce_bool, _split_known


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
