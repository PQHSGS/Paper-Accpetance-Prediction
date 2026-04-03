import re
from typing import TYPE_CHECKING, Any

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from ..feature.science_parse_features import get_paper_content

if TYPE_CHECKING:
    from ..config.feature import FeatureConfig
    from ..config.preprocess import PreprocessConfig
    from ..entities import ProcessData

# Cache stopwords once at module level (avoids reloading from disk per call)
_STOP_WORDS = set(stopwords.words("english"))
_TOKENIZER = RegexpTokenizer(r"\w+")


def normalize_text(text: str, only_char: bool = False, lower: bool = False, stop_remove: bool = False) -> str:
    """
    Normalizes the input text by filtering out non-ASCII, lowercasing, and removing stopwords.

    Args:
        text: Input text to normalize.
        only_char: If True, retains only alphanumeric characters.
        lower: If True, converts the text to lowercase.
        stop_remove: If True, removes English stopwords.

    Returns:
        Normalized text string.
    """
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    if lower:
        text = text.lower()

    if only_char:
        tokens = _TOKENIZER.tokenize(text)
    else:
        tokens = text.split()

    if stop_remove:
        tokens = [w for w in tokens if w not in _STOP_WORDS]

    # Remove single-character tokens.
    tokens = [w for w in tokens if len(w) > 1]
    return " ".join(tokens)


def build_corpus_words(
    data: "ProcessData",
    preprocess_config: "PreprocessConfig",
    feature_config: "FeatureConfig",
    context: dict[str, Any],
) -> "ProcessData":
    """Build or load normalized corpus words and attach them to ProcessData metadata."""
    import os
    import pickle as pkl

    cache_dir = context["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    corpus_path = os.path.join(cache_dir, "corpus.pkl")

    if preprocess_config.preserve_corpus_cache and os.path.exists(corpus_path):
        with open(corpus_path, "rb") as f:
            corpus_words = pkl.load(f)
    else:
        corpus_words = []
        for paper in data.papers:
            content = get_paper_content(paper.SCIENCEPARSE)
            norm = normalize_text(
                content,
                only_char=preprocess_config.only_char,
                lower=preprocess_config.lower,
                stop_remove=preprocess_config.stop_remove,
            )
            corpus_words.extend(norm.split(" "))
        with open(corpus_path, "wb") as f:
            pkl.dump(corpus_words, f)

    data.metadata["corpus_words"] = corpus_words
    return data


def compute_frequency_buckets(
    data: "ProcessData",
    preprocess_config: "PreprocessConfig",
    feature_config: "FeatureConfig",
    context: dict[str, Any],
) -> "ProcessData":
    """Compute high/moderate/low-frequency word buckets from corpus words."""
    from ..feature.handcrafted import count_words

    _ = feature_config  # kept for future extensibility in signature parity.
    _ = context
    corpus_words = data.metadata.get("corpus_words", [])

    data.hfws, data.most_frequent_words, data.least_frequent_words = count_words(
        corpus_words,
        preprocess_config.hfw_proportion,
        preprocess_config.freq_proportion,
        preprocess_config.min_freq_threshold,
    )
    return data
