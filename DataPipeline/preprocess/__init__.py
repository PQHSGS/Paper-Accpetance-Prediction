from typing import Callable, List

from .normalization import build_corpus_words, compute_frequency_buckets, normalize_text
from .parsing import load_papers_from_dir, read_science_parse
from .feature_transform import standardize_for_linear_models

DATA_PREPROCESS_METHODS: dict[str, Callable] = {
    "build_corpus_words": build_corpus_words,
    "compute_frequency_buckets": compute_frequency_buckets,
}

FEATURE_PREPROCESS_METHODS: dict[str, Callable] = {
    "standardize_for_linear_models": standardize_for_linear_models,
}


def get_data_preprocess_methods(method_names: List[str]) -> List[Callable]:
    methods: List[Callable] = []
    for name in method_names:
        if name not in DATA_PREPROCESS_METHODS:
            raise ValueError(
                f"Unknown preprocess method {name!r}. Available methods: {sorted(DATA_PREPROCESS_METHODS)}"
            )
        methods.append(DATA_PREPROCESS_METHODS[name])
    return methods


def get_feature_preprocess_methods(method_names: List[str]) -> List[Callable]:
    methods: List[Callable] = []
    for name in method_names:
        if name not in FEATURE_PREPROCESS_METHODS:
            raise ValueError(
                f"Unknown feature preprocess method {name!r}. "
                f"Available methods: {sorted(FEATURE_PREPROCESS_METHODS)}"
            )
        methods.append(FEATURE_PREPROCESS_METHODS[name])
    return methods


__all__ = [
	"normalize_text",
	"build_corpus_words",
	"compute_frequency_buckets",
	"load_papers_from_dir",
    "read_science_parse",
	"DATA_PREPROCESS_METHODS",
    "FEATURE_PREPROCESS_METHODS",
	"get_data_preprocess_methods",
    "get_feature_preprocess_methods",
    "standardize_for_linear_models",
]
