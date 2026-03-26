from typing import Tuple, Set, List, Dict, Any
from collections import Counter

def count_words(corpus: List[str], hfw_proportion: float, freq_proportion: float, min_freq_thr: int) -> Tuple[List[str], Set[str], Set[str]]:
    """
    Counts words in a corpus and returns high-frequency words, 
    moderate-frequency words, and least-frequent words.
    """
    counter = Counter(corpus)
    n_hfw = int(len(counter) * hfw_proportion)
    n_combined = int(len(counter) * (hfw_proportion + freq_proportion))
    
    # Single most_common call with the larger limit
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
        paper (Any): The instantiated Paper model.
        science_parse (Any): The ScienceParse model.
        hfws (List[str]): High frequency words list.
        freq_words (Set[str]): Moderate frequency words.
        infreq_words (Set[str]): Low frequency words.
        
    Returns:
        Dict[str, float]: Mapping of feature names to values.
    """
    features = {
        "get_most_recent_reference_year": science_parse.get_most_recent_reference_year() - 2000,
        "get_num_references": science_parse.get_num_references(),
        "get_num_refmentions": science_parse.get_num_refmentions(),
        "get_avg_length_reference_mention_contexts": science_parse.get_avg_length_reference_mention_contexts(),
        "abstract_contains_deep": int(paper.abstract_contains_a_term("deep")),
        "abstract_contains_neural": int(paper.abstract_contains_a_term("neural")),
        "abstract_contains_embedding": int(paper.abstract_contains_a_term("embedding")),
        "abstract_contains_outperform": int(paper.abstract_contains_a_term("outperform")),
        "abstract_contains_novel": int(paper.abstract_contains_a_term("novel")),
        "abstract_contains_state_of_the_art": int(paper.abstract_contains_a_term("state of the art")),
        "abstract_contains_state-of-the-art": int(paper.abstract_contains_a_term("state-of-the-art")),
        "get_num_recent_references": science_parse.get_num_recent_references(2017),
        "get_num_ref_to_figures": science_parse.get_num_ref_to_figures(),
        "get_num_ref_to_tables": science_parse.get_num_ref_to_tables(),
        "get_num_ref_to_sections": science_parse.get_num_ref_to_sections(),
        "get_num_uniq_words": science_parse.get_num_uniq_words(),
        "get_num_sections": science_parse.get_num_sections(),
        "get_avg_sentence_length": science_parse.get_avg_sentence_length(),
        "get_contains_appendix": science_parse.get_contains_appendix(),
        "proportion_of_frequent_words": round(science_parse.get_frequent_words_proportion(hfws, freq_words, infreq_words), 3),
        "get_title_length": paper.get_title_len(),
        "get_num_authors": science_parse.get_num_authors(),
        "get_num_ref_to_equations": science_parse.get_num_ref_to_equations(),
        "get_num_ref_to_theorems": science_parse.get_num_ref_to_theorems()
    }
    return features
