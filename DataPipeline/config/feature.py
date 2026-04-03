from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .common import _coerce_bool, _coerce_bool_or_int, _coerce_str_list, _split_known


@dataclass
class FeatureConfig:
    methods: List[str] = field(default_factory=lambda: ["handcrafted_features"])
    max_vocab: Union[bool, int] = False
    encoder_type: Union[bool, str] = "w2v"
    use_hand_features: bool = True
    drop_post_review_leakage: bool = True
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
            "methods",
            "max_vocab",
            "encoder_type",
            "use_hand_features",
            "drop_post_review_leakage",
        }
        known, extra = _split_known(data, known_keys)

        if "methods" in known:
            known["methods"] = _coerce_str_list(known["methods"], "FeatureConfig.methods")

        if "max_vocab" in known:
            known["max_vocab"] = _coerce_bool_or_int(known["max_vocab"])
        for bool_key in ("use_hand_features", "drop_post_review_leakage"):
            if bool_key in known:
                known[bool_key] = _coerce_bool(known[bool_key])

        cfg = cls(**known)
        cfg.extra = extra
        return cfg
