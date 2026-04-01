from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

import numpy as np

from .science_parse_features import (
    get_avg_length_reference_mention_contexts,
    get_avg_sentence_length,
    get_contains_appendix,
    get_content_words,
    get_frequent_words_proportion,
    get_num_authors,
    get_num_recent_references,
    get_num_ref_to_equations,
    get_num_ref_to_figures,
    get_num_ref_to_sections,
    get_num_ref_to_tables,
    get_num_ref_to_theorems,
    get_num_references,
    get_num_refmentions,
    get_num_sections,
    get_num_uniq_words,
    get_reference_years_dict,
    get_sections_dict,
)

if TYPE_CHECKING:
    from ..config.feature import FeatureConfig
    from ..entities import ProcessData


def count_words(corpus: List[str], hfw_proportion: float, freq_proportion: float, min_freq_thr: int) -> Tuple[List[str], Set[str], Set[str]]:
    """
    Counts words in a corpus and returns high-frequency words,
    moderate-frequency words, and least-frequent words.
    """
    counter = Counter(corpus)
    n_hfw = int(len(counter) * hfw_proportion)
    n_combined = int(len(counter) * (hfw_proportion + freq_proportion))

    # Single most_common call with the larger limit.
    top_combined = counter.most_common(n_combined)
    most_common_set = set(x[0] for x in top_combined[:n_hfw])
    most_common = [x[0] for x in top_combined[:n_hfw]]
    most_common2_set = set(x[0] for x in top_combined)

    most_frequent_words = most_common2_set - most_common_set
    least_frequent_words = set(w for w, c in counter.items() if c < min_freq_thr)

    return most_common, most_frequent_words, least_frequent_words


def extract_hand_features(paper: Any, science_parse: Any, hfws: List[str], freq_words: Set[str], infreq_words: Set[str]) -> Dict[str, float]:
    """
    Extracts hand-crafted features from a paper and its parsed contents using the dictionary format.

    Args:
        paper: The instantiated Paper model.
        science_parse: The ScienceParse model.
        hfws: High frequency words list.
        freq_words: Moderate frequency words.
        infreq_words: Low frequency words.

    Returns:
        Mapping of feature names to values.
    """
    num_references = get_num_references(science_parse)
    num_refmentions = get_num_refmentions(science_parse)
    num_sections = get_num_sections(science_parse)
    num_uniq_words = get_num_uniq_words(science_parse)
    total_words = max(1, len(get_content_words(science_parse)))

    num_recent_references = get_num_recent_references(science_parse, 2017)
    num_ref_to_figures = get_num_ref_to_figures(science_parse)
    num_ref_to_tables = get_num_ref_to_tables(science_parse)
    num_ref_to_sections = get_num_ref_to_sections(science_parse)
    num_ref_to_equations = get_num_ref_to_equations(science_parse)
    num_ref_to_theorems = get_num_ref_to_theorems(science_parse)

    reference_years = [
        y for y in get_reference_years_dict(science_parse).values()
        if isinstance(y, int) and 1900 <= y <= 2025
    ]
    if reference_years:
        ref_ages = [2017 - y for y in reference_years]
        avg_reference_age = float(np.mean(ref_ages))
        median_reference_age = float(np.median(ref_ages))
        frac_recent_refs = float(sum(1 for y in reference_years if 2017 - y < 5) / len(reference_years))
    else:
        avg_reference_age = 0.0
        median_reference_age = 0.0
        frac_recent_refs = 0.0

    section_titles = [str(k).lower() for k in get_sections_dict(science_parse).keys()]
    has_related_work = int(any("related" in s and "work" in s for s in section_titles))
    has_experiments = int(any("experiment" in s or "evaluation" in s for s in section_titles))
    has_conclusion = int(any("conclusion" in s for s in section_titles))

    # Merge state-of-the-art duplicates into one robust flag.
    contains_sota = int(
        paper.abstract_contains_a_term("state of the art")
        or paper.abstract_contains_a_term("state-of-the-art")
    )

    features = {
        # Core paper/reference signals.
        "get_num_references": num_references,
        "get_num_refmentions": num_refmentions,
        "get_avg_length_reference_mention_contexts": get_avg_length_reference_mention_contexts(science_parse),
        "get_num_recent_references": num_recent_references,
        "get_num_ref_to_figures": num_ref_to_figures,
        "get_num_ref_to_tables": num_ref_to_tables,
        "get_num_ref_to_sections": num_ref_to_sections,
        "get_num_ref_to_equations": num_ref_to_equations,
        "get_num_ref_to_theorems": num_ref_to_theorems,
        # Raw structure/lexical features.
        "get_num_uniq_words": num_uniq_words,
        "get_num_sections": num_sections,
        "get_avg_sentence_length": get_avg_sentence_length(science_parse),
        "get_contains_appendix": get_contains_appendix(science_parse),
        "proportion_of_frequent_words": round(get_frequent_words_proportion(science_parse, hfws, freq_words, infreq_words), 3),
        "get_title_length": paper.get_title_len(),
        "get_num_authors": get_num_authors(science_parse),
        # Robust keyword indicators.
        "abstract_contains_deep": int(paper.abstract_contains_a_term("deep")),
        "abstract_contains_neural": int(paper.abstract_contains_a_term("neural")),
        "abstract_contains_embedding": int(paper.abstract_contains_a_term("embedding")),
        "abstract_contains_outperform": int(paper.abstract_contains_a_term("outperform")),
        "abstract_contains_novel": int(paper.abstract_contains_a_term("novel")),
        "abstract_contains_state_of_the_art": contains_sota,
        # Advanced normalized features.
        "refs_per_section": num_references / max(1, num_sections),
        "refmentions_per_reference": num_refmentions / max(1, num_references),
        "figures_per_section": num_ref_to_figures / max(1, num_sections),
        "tables_per_section": num_ref_to_tables / max(1, num_sections),
        "equations_per_section": num_ref_to_equations / max(1, num_sections),
        "section_mentions_per_section": num_ref_to_sections / max(1, num_sections),
        "unique_word_ratio": num_uniq_words / total_words,
        "recent_ref_ratio": num_recent_references / max(1, num_references),
        # Advanced citation-age features.
        "avg_reference_age": avg_reference_age,
        "median_reference_age": median_reference_age,
        "frac_recent_refs": frac_recent_refs,
        # Section-presence features.
        "has_related_work_section": has_related_work,
        "has_experiments_section": has_experiments,
        "has_conclusion_section": has_conclusion,
        # Interaction feature.
        "recent_refs_x_novel": num_recent_references * int(paper.abstract_contains_a_term("novel")),
    }
    return features


def handcrafted_features(
    paper: Any,
    process_data: "ProcessData",
    feature_config: "FeatureConfig",
) -> Dict[str, float]:
    """Registry-friendly wrapper for the current handcrafted feature extractor."""
    _ = feature_config
    return extract_hand_features(
        paper,
        paper.SCIENCEPARSE,
        process_data.hfws,
        process_data.most_frequent_words,
        process_data.least_frequent_words,
    )
