import re
from typing import Any, Dict, List, Sequence, Set


def get_sections_dict(science_parse: Any) -> Dict[str, str]:
    return science_parse.sections


def get_reference_years_dict(science_parse: Any) -> Dict[int, int]:
    return science_parse.reference_years


def get_reference_mention_contexts_dict(science_parse: Any) -> Dict[int, str]:
    return science_parse.reference_mention_contexts


def get_reference_num_mentions_dict(science_parse: Any) -> Dict[int, int]:
    return science_parse.reference_num_mentions


def get_num_references(science_parse: Any) -> int:
    return len(get_reference_years_dict(science_parse))


def get_num_refmentions(science_parse: Any) -> int:
    return int(sum(get_reference_num_mentions_dict(science_parse).values()))


def get_avg_length_reference_mention_contexts(science_parse: Any) -> float:
    contexts = get_reference_mention_contexts_dict(science_parse)
    if not contexts:
        return 0.0
    return float(sum(len(v) for v in contexts.values()) / len(contexts))


def get_paper_content(science_parse: Any) -> str:
    cached = getattr(science_parse, "_cached_content", None)
    if cached is not None:
        return cached

    content = (
        science_parse.title
        + " "
        + science_parse.abstract
        + " "
        + get_author_names_string(science_parse)
        + " "
        + get_domains_from_emails(science_parse)
    )
    for sect_id in sorted(science_parse.sections):
        content = content + " " + science_parse.sections[sect_id]
    content = re.sub("\n([0-9]*\n)+", "\n", content)
    science_parse._cached_content = content
    return content


def get_content_words(science_parse: Any) -> List[str]:
    cached = getattr(science_parse, "_cached_content_words", None)
    if cached is not None:
        return cached
    words = get_paper_content(science_parse).split(" ")
    science_parse._cached_content_words = words
    return words


def get_frequent_words_proportion(
    science_parse: Any,
    hfws: Sequence[str],
    most_frequent_words: Set[str],
    least_frequent_words: Set[str],
) -> float:
    content = get_content_words(science_parse)
    n = 0
    t = 0
    for w in content:
        if w not in hfws and w not in least_frequent_words:
            t += 1
            n += int(w in most_frequent_words)
    if t == 0:
        return 0.0
    return float(n / t)


def get_num_recent_references(science_parse: Any, submission_year: int) -> int:
    years = get_reference_years_dict(science_parse)
    return int(sum(1 for y in years.values() if submission_year - y < 5))


def get_num_ref_to_figures(science_parse: Any) -> int:
    return int(sum(1 for x in get_content_words(science_parse) if x == "Figure"))


def get_num_ref_to_tables(science_parse: Any) -> int:
    return int(sum(1 for x in get_content_words(science_parse) if x == "Table"))


def get_num_ref_to_sections(science_parse: Any) -> int:
    return int(sum(1 for x in get_content_words(science_parse) if x == "Section"))


def get_num_uniq_words(science_parse: Any) -> int:
    return len(set(get_content_words(science_parse)))


def get_num_sections(science_parse: Any) -> int:
    return len(science_parse.sections)


def get_avg_sentence_length(science_parse: Any) -> float:
    sentences = get_paper_content(science_parse).split(". ")
    lengths = [len(s.split(" ")) for s in sentences if s]
    if not lengths:
        return 0.0
    return float(sum(lengths) / len(lengths))


def get_contains_appendix(science_parse: Any) -> int:
    return int("Appendix" in get_content_words(science_parse))


def get_num_authors(science_parse: Any) -> int:
    if science_parse.authors is None:
        return 0
    return len(science_parse.authors)


def get_author_names_string(science_parse: Any) -> str:
    if science_parse.authors is None:
        return ""
    return str.join(" ", science_parse.authors)


def get_domains_from_emails(science_parse: Any) -> str:
    if science_parse.emails is None:
        return ""
    domains = []
    for email in science_parse.emails:
        if "@" in email:
            domains.append(email.split("@")[1].replace(".", "_"))
    return str.join(" ", domains)


def get_num_ref_to_equations(science_parse: Any) -> int:
    return int(sum(1 for x in get_content_words(science_parse) if x == "Equation"))


def get_num_ref_to_theorems(science_parse: Any) -> int:
    return int(sum(1 for x in get_content_words(science_parse) if x == "Theorem"))
