import os
import sys
import pickle as pkl
import random
import time
from typing import Optional, Set, Dict, List, Tuple, Any
# Add parent path to support models package
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from parsing import load_papers_from_dir
from normalization import normalize_text
from feature_extraction import count_words, extract_hand_features
from reporting import read_features, save_features_to_file


# Features directly tied to reviewer scores/recommendations are considered
# post-review acceptance proxies and are excluded to avoid leakage.
_POST_REVIEW_LEAKAGE_TOKENS = {
    "recommendation",
    "review_score",
    "overall_score",
    "confidence",
}


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_accepted(accepted: Any) -> Optional[bool]:
    if isinstance(accepted, bool):
        return accepted
    if isinstance(accepted, int):
        if accepted in (0, 1):
            return bool(accepted)
    if isinstance(accepted, str):
        v = accepted.strip().lower()
        if v in {"true", "1", "accept", "accepted", "yes"}:
            return True
        if v in {"false", "0", "reject", "rejected", "no"}:
            return False
    return None


def _get_avg_recommendation(paper) -> Optional[float]:
    recs = []
    for review in paper.get_reviews():
        rec = review.get_recommendation()
        try:
            if rec is not None and str(rec).strip() != "":
                recs.append(float(rec))
        except Exception:
            continue

    if not recs:
        return None
    return float(sum(recs) / len(recs))


def _resolve_accept_label(
    paper,
    accepted_titles: Set[str],
    allow_recommendation_fallback: bool,
) -> tuple[int, str]:
    """
    Resolve paper label with dataset-aware fallback order.

    Priority:
    1) Explicit `accepted` field in review JSON.
    2) ACL title list (`acl_accepted.txt`) when available.
    3) Average reviewer recommendation >= 3.5 (used for datasets where
       explicit labels are unavailable in released JSON, e.g., CoNLL).
    """
    accepted = _coerce_accepted(paper.get_accepted())
    if accepted is not None:
        return int(accepted), "json.accepted"

    if accepted_titles:
        label = int(paper.get_title().strip().lower() in accepted_titles)
        return label, "acl_accepted.txt"

    if allow_recommendation_fallback:
        avg_rec = _get_avg_recommendation(paper)
        if avg_rec is not None:
            # Matches PeerRead CoNLL distribution (11/11) on released splits.
            return int(avg_rec >= 3.5), "review.recommendation>=3.5"

    raise ValueError(
        f"Could not resolve label for paper ID={paper.get_id()} title={paper.get_title()!r}. "
        "No explicit `accepted` field and no accepted-title mapping."
    )


def _is_post_review_leakage_feature(feature_name: str) -> bool:
    lname = feature_name.lower()
    return any(tok in lname for tok in _POST_REVIEW_LEAKAGE_TOKENS)


def _load_accepted_titles(dataset_root: str) -> Set[str]:
    accepted_txt_path = os.path.join(dataset_root, 'acl_accepted.txt')
    accepted_titles: Set[str] = set()
    if os.path.exists(accepted_txt_path):
        with open(accepted_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                t = line.strip().lower()
                if t:
                    accepted_titles.add(t)
    return accepted_titles


def _label_dataset_papers(
    raw_dataset_papers: List[Any],
    accepted_titles: Set[str],
    allow_recommendation_fallback: bool,
) -> Tuple[List[Any], Dict[str, int], int]:
    labeled: List[Any] = []
    skipped_unlabeled = 0
    label_src_counts = {
        "json.accepted": 0,
        "acl_accepted.txt": 0,
        "review.recommendation>=3.5": 0,
    }

    for p in raw_dataset_papers:
        try:
            label, src = _resolve_accept_label(
                p,
                accepted_titles,
                allow_recommendation_fallback=allow_recommendation_fallback,
            )
        except ValueError:
            skipped_unlabeled += 1
            continue

        p.ACCEPTED = bool(label)
        label_src_counts[src] += 1
        labeled.append(p)

    return labeled, label_src_counts, skipped_unlabeled


def _normalize_corpus_words(out_dir: str, papers: List[Any]) -> List[str]:
    paper_content_corpus = []
    for paper in papers:
        content = paper.SCIENCEPARSE.get_paper_content()
        paper_content_corpus.append(normalize_text(content, only_char=True, lower=True, stop_remove=True))

    out_corpus_filename = os.path.join(out_dir, 'corpus.pkl')
    if not os.path.isfile(out_corpus_filename):
        corpus_words: List[str] = []
        for p_content in paper_content_corpus:
            corpus_words += p_content.split(' ')
        pkl.dump(corpus_words, open(out_corpus_filename, 'wb'))
    else:
        corpus_words = pkl.load(open(out_corpus_filename, 'rb'))

    return corpus_words

def main(args):
    if len(args) < 9:
        print("Usage:", args[0], "<paper-json-dir> <scienceparse-dir> <out-dir> <submission-year> <feature_output_file> <tfidf_vector_file> <max_vocab_size> <encoder> <hand-feature>")
        return -1

    paper_json_dirs = args[1].split(',')
    scienceparse_dirs = args[2].split(',')
    out_dir = args[3]
    feature_output_file = args[4]
    vect_file = args[5]
    max_vocab_size = False if args[6] == 'False' else int(args[6])
    encoder_type = False if args[7] == 'False' else str(args[7])
    use_hand = False if args[8] == 'False' else True
    # Default True to preserve full-sample behavior for this repository's datasets.
    # Set ALLOW_RECOMMENDATION_LABEL_FALLBACK=False for strict no-score labeling.
    allow_recommendation_fallback = _parse_bool_env("ALLOW_RECOMMENDATION_LABEL_FALLBACK", default=True)
    shuffle_seed = int(os.getenv("EXTRACTION_SHUFFLE_SEED", "42"))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    is_train = not os.path.isfile(feature_output_file)
    idToFeature = dict()

    if is_train:
        print('Loading vector file from scratch..')
    else:
        print('Loading features and vector file from existing...')
        idToFeature = read_features(feature_output_file)
        # Vector loading logic would go here

    # 1. Parsing Module
    t_start = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] Reading reviews and science parse from {len(paper_json_dirs)} directories...')
    papers = []
    
    for idx, (p_dir, s_dir) in enumerate(zip(paper_json_dirs, scienceparse_dirs)):
        t_ds = time.time()
        raw_dataset_papers = load_papers_from_dir(p_dir, s_dir)
        dataset_papers = []
        dataset_root = os.path.dirname(os.path.dirname(p_dir))
        accepted_titles = _load_accepted_titles(dataset_root)

        dataset_papers, label_src_counts, skipped_unlabeled = _label_dataset_papers(
            raw_dataset_papers=raw_dataset_papers,
            accepted_titles=accepted_titles,
            allow_recommendation_fallback=allow_recommendation_fallback,
        )
                
        papers.extend(dataset_papers)
        print(
            f'  [{time.strftime("%H:%M:%S")}] Dataset {idx+1}/{len(paper_json_dirs)}: {p_dir} '
            f'-> {len(dataset_papers)}/{len(raw_dataset_papers)} papers ({time.time()-t_ds:.1f}s) | '
            f'label_source={label_src_counts} | skipped_unlabeled={skipped_unlabeled}'
        )
        
    rng = random.Random(shuffle_seed)
    rng.shuffle(papers)
    print(f'[{time.strftime("%H:%M:%S")}] Total papers loaded: {len(papers)} ({time.time()-t_start:.1f}s)')

    def get_feature_id(feature_name: str) -> int:
        return idToFeature.get(feature_name)

    def add_feature_to_dict(fname: str) -> int:
        if fname not in idToFeature:
            fid = len(idToFeature)
            idToFeature[fname] = fid
        return idToFeature[fname]

    # 2. Normalization Module
    t_norm = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] Normalizing {len(papers)} papers...')
    paper_content_corpus = []
    for i, paper in enumerate(papers):
        content = paper.SCIENCEPARSE.get_paper_content()
        paper_content_corpus.append(normalize_text(content, only_char=True, lower=True, stop_remove=True))
        if (i+1) % 500 == 0 or (i+1) == len(papers):
            print(f'  [{time.strftime("%H:%M:%S")}] Normalized {i+1}/{len(papers)} papers ({time.time()-t_norm:.1f}s)')

    # Reuse previous corpus cache when present to preserve behavior.
    outCorpusFilename = os.path.join(out_dir, 'corpus.pkl')
    if not os.path.isfile(outCorpusFilename):
        paper_content_corpus_words = []
        for p_content in paper_content_corpus:
            paper_content_corpus_words += p_content.split(' ')
        pkl.dump(paper_content_corpus_words, open(outCorpusFilename, 'wb'))
    else:
        paper_content_corpus_words = pkl.load(open(outCorpusFilename, 'rb'))
    print(f'[{time.strftime("%H:%M:%S")}] Total words in normalized corpus: {len(paper_content_corpus_words)} ({time.time()-t_norm:.1f}s)')

    # 3. Feature extraction Module
    t_feat = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] Counting words and extracting features...')
    hfws, most_frequent_words, least_frequent_words = count_words(paper_content_corpus_words, 0.01, 0.05, 3)
    dropped_leakage_features = set()

    outLabelsFile = open(os.path.join(out_dir, f'labels_{max_vocab_size}_{encoder_type}_{use_hand}.tsv'), 'w')
    outIDFile = open(os.path.join(out_dir, f'ids_{max_vocab_size}_{encoder_type}_{use_hand}.tsv'), 'w')
    outSvmLiteFile = open(os.path.join(out_dir, f'features.svmlite_{max_vocab_size}_{encoder_type}_{use_hand}.txt'), 'w')

    print(f'[{time.strftime("%H:%M:%S")}] Writing SVMLite features for {len(papers)} papers...')
    for p_idx, paper in enumerate(papers, start=1):
        outIDFile.write(f"{p_idx}\t{paper.get_title()}\n")
        label = int(paper.get_accepted() is True)
            
        outLabelsFile.write(str(label) + "\n")

        # SVMLite Export
        combined_features = {}
        if use_hand:
            hand_features = extract_hand_features(paper, paper.SCIENCEPARSE, hfws, most_frequent_words, least_frequent_words)
            for feature_name, value in hand_features.items():
                if _is_post_review_leakage_feature(feature_name):
                    dropped_leakage_features.add(feature_name)
                    continue

                if is_train:
                    fid = add_feature_to_dict(feature_name)
                else:
                    fid = get_feature_id(feature_name)
                
                if fid is not None and value != 0:
                    combined_features[fid] = value

        # Writing out row for SVMLite
        outSvmLiteFile.write(str(label) + " ")
        for fid in sorted(combined_features.keys()):
            outSvmLiteFile.write(f"{fid}:{combined_features[fid]} ")
        outSvmLiteFile.write("\n")
        
        if p_idx % 500 == 0 or p_idx == len(papers):
            print(f'  [{time.strftime("%H:%M:%S")}] Exported {p_idx}/{len(papers)} papers ({time.time()-t_feat:.1f}s)')

    if is_train:
        save_features_to_file(idToFeature, feature_output_file)

    if dropped_leakage_features:
        print(f"[{time.strftime('%H:%M:%S')}] Dropped leakage features: {sorted(dropped_leakage_features)}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] No post-review score features found in exported columns.")

    outLabelsFile.close()
    outIDFile.close()
    outSvmLiteFile.close()

    # Safety check: sample count in labels must equal papers processed.
    labels_path = os.path.join(out_dir, f'labels_{max_vocab_size}_{encoder_type}_{use_hand}.tsv')
    with open(labels_path, 'r', encoding='utf-8') as lf:
        n_labels = sum(1 for _ in lf)
    if n_labels != len(papers):
        raise RuntimeError(
            f"Label count mismatch: expected {len(papers)} rows but found {n_labels} in {labels_path}."
        )

    total_time = time.time() - t_start
    print(f'[{time.strftime("%H:%M:%S")}] Done! Saved feature pipelines properly. Total time: {total_time:.1f}s')

if __name__ == "__main__":
    main(sys.argv)
