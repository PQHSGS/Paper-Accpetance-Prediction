from typing import Callable, List

from .feature_extraction import count_words, extract_hand_features, handcrafted_features
from .reporting import read_features, save_features_to_file, write_svmlite_row

FEATURE_METHODS: dict[str, Callable] = {
    "handcrafted_features": handcrafted_features,
}


def get_feature_methods(method_names: List[str]) -> List[Callable]:
    methods: List[Callable] = []
    for name in method_names:
        if name not in FEATURE_METHODS:
            raise ValueError(
                f"Unknown feature method {name!r}. Available methods: {sorted(FEATURE_METHODS)}"
            )
        methods.append(FEATURE_METHODS[name])
    return methods


__all__ = [
	"count_words",
	"extract_hand_features",
	"handcrafted_features",
	"read_features",
	"save_features_to_file",
	"write_svmlite_row",
	"FEATURE_METHODS",
	"get_feature_methods",
]
