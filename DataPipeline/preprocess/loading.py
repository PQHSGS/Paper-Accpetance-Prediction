import os
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from ..entities import ProcessData
from .parsing import load_papers_from_dir

if TYPE_CHECKING:
    from ..config.pipeline import PipelineConfig


class LabelResolver:
    def __init__(self, allow_recommendation_fallback: bool, recommendation_threshold: float):
        self.allow_recommendation_fallback = allow_recommendation_fallback
        self.recommendation_threshold = recommendation_threshold

    @staticmethod
    def _coerce_accepted(accepted: Any) -> Optional[bool]:
        if isinstance(accepted, bool):
            return accepted
        if isinstance(accepted, int) and accepted in (0, 1):
            return bool(accepted)
        if isinstance(accepted, str):
            v = accepted.strip().lower()
            if v in {"true", "1", "accept", "accepted", "yes"}:
                return True
            if v in {"false", "0", "reject", "rejected", "no"}:
                return False
        return None

    @staticmethod
    def _get_avg_recommendation(paper: Any) -> Optional[float]:
        recs: List[float] = []
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

    @staticmethod
    def load_accepted_titles(dataset_root: str) -> Set[str]:
        accepted_txt_path = os.path.join(dataset_root, "acl_accepted.txt")
        accepted_titles: Set[str] = set()
        if os.path.exists(accepted_txt_path):
            with open(accepted_txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip().lower()
                    if t:
                        accepted_titles.add(t)
        return accepted_titles

    def resolve_label(self, paper: Any, accepted_titles: Set[str]) -> Tuple[int, str]:
        accepted = self._coerce_accepted(paper.get_accepted())
        if accepted is not None:
            return int(accepted), "json.accepted"

        if accepted_titles:
            label = int(paper.get_title().strip().lower() in accepted_titles)
            return label, "acl_accepted.txt"

        if self.allow_recommendation_fallback:
            avg_rec = self._get_avg_recommendation(paper)
            if avg_rec is not None:
                return int(avg_rec >= self.recommendation_threshold), "review.recommendation>=threshold"

        raise ValueError(
            f"Could not resolve label for paper ID={paper.get_id()} title={paper.get_title()!r}. "
            "No explicit accepted label and no accepted-title mapping."
        )


def load_split_data(config: "PipelineConfig", split: str) -> ProcessData:
    papers: List[Any] = []
    label_source_counts = {
        "json.accepted": 0,
        "acl_accepted.txt": 0,
        "review.recommendation>=threshold": 0,
    }
    skipped_unlabeled = 0

    resolver = LabelResolver(
        allow_recommendation_fallback=config.feature.allow_recommendation_fallback,
        recommendation_threshold=config.feature.recommendation_threshold,
    )

    data_cfg = config.data
    print(f"Loading split={split} from {len(data_cfg.datasets)} datasets...")
    for idx, dataset_name in enumerate(data_cfg.datasets, start=1):
        p_dir = os.path.join(data_cfg.base_dir, dataset_name, split, data_cfg.reviews_subdir)
        s_dir = os.path.join(data_cfg.base_dir, dataset_name, split, data_cfg.parsed_subdir)

        if not (os.path.isdir(p_dir) and os.path.isdir(s_dir)):
            continue

        raw_dataset_papers = load_papers_from_dir(p_dir, s_dir)
        accepted_titles = resolver.load_accepted_titles(os.path.join(data_cfg.base_dir, dataset_name))

        dataset_papers: List[Any] = []
        dataset_label_src_counts = {
            "json.accepted": 0,
            "acl_accepted.txt": 0,
            "review.recommendation>=threshold": 0,
        }
        dataset_skipped = 0
        for p in raw_dataset_papers:
            try:
                label, src = resolver.resolve_label(p, accepted_titles)
            except ValueError:
                dataset_skipped += 1
                continue

            p.ACCEPTED = bool(label)
            dataset_label_src_counts[src] += 1
            dataset_papers.append(p)

        papers.extend(dataset_papers)
        skipped_unlabeled += dataset_skipped
        for key, value in dataset_label_src_counts.items():
            label_source_counts[key] += value

        print(
            f"  Dataset {idx}/{len(data_cfg.datasets)} {dataset_name}: "
            f"{len(dataset_papers)}/{len(raw_dataset_papers)} papers | "
            f"label_source={dataset_label_src_counts} | skipped_unlabeled={dataset_skipped}"
        )

    rng = np.random.default_rng(config.feature.shuffle_seed)
    if papers:
        order = rng.permutation(len(papers))
        papers = [papers[i] for i in order]

    labels = [int(p.get_accepted() is True) for p in papers]
    titles = [p.get_title() for p in papers]

    print(f"Loaded {len(papers)} labeled papers for split={split}.")
    return ProcessData(
        split=split,
        papers=papers,
        labels=labels,
        titles=titles,
        label_source_counts=label_source_counts,
        skipped_unlabeled=skipped_unlabeled,
        metadata={"n_papers": len(papers)},
    )
