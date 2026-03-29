from typing import Callable, List

from .normalization import build_corpus_words, compute_frequency_buckets, normalize_text
from .parsing import load_papers_from_dir
from .science_parse_reader import ScienceParseReader

PREPROCESS_METHODS: dict[str, Callable] = {
    "build_corpus_words": build_corpus_words,
    "compute_frequency_buckets": compute_frequency_buckets,
}


def get_preprocess_methods(method_names: List[str]) -> List[Callable]:
    methods: List[Callable] = []
    for name in method_names:
        if name not in PREPROCESS_METHODS:
            raise ValueError(
                f"Unknown preprocess method {name!r}. Available methods: {sorted(PREPROCESS_METHODS)}"
            )
        methods.append(PREPROCESS_METHODS[name])
    return methods


__all__ = [
	"normalize_text",
	"build_corpus_words",
	"compute_frequency_buckets",
	"load_papers_from_dir",
    "ScienceParseReader",
	"PREPROCESS_METHODS",
	"get_preprocess_methods",
]
