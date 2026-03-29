import os
import pickle as pkl
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

try:
    from .config import Config
    from .feature_extraction import count_words, extract_hand_features
    from .normalization import normalize_text
    from .parsing import load_papers_from_dir
    from .reporting import read_features, save_features_to_file, write_svmlite_row
except ImportError:
    from config import Config
    from feature_extraction import count_words, extract_hand_features
    from normalization import normalize_text
    from parsing import load_papers_from_dir
    from reporting import read_features, save_features_to_file, write_svmlite_row
from Models import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVMClassifier,
)


_POST_REVIEW_LEAKAGE_TOKENS = {
    "recommendation",
    "review_score",
    "overall_score",
    "confidence",
}

_MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "SVMClassifier": SVMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


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

    def label_dataset(self, raw_dataset_papers: Sequence[Any], accepted_titles: Set[str]) -> Tuple[List[Any], Dict[str, int], int]:
        labeled: List[Any] = []
        skipped_unlabeled = 0
        label_src_counts = {
            "json.accepted": 0,
            "acl_accepted.txt": 0,
            "review.recommendation>=threshold": 0,
        }

        for p in raw_dataset_papers:
            try:
                label, src = self.resolve_label(p, accepted_titles)
            except ValueError:
                skipped_unlabeled += 1
                continue

            p.ACCEPTED = bool(label)
            label_src_counts[src] += 1
            labeled.append(p)

        return labeled, label_src_counts, skipped_unlabeled


class FeaturePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.label_resolver = LabelResolver(
            allow_recommendation_fallback=self.config.feature.allow_recommendation_fallback,
            recommendation_threshold=self.config.feature.recommendation_threshold,
        )
        self.id_to_feature: Dict[str, int] = {}
        self.hfws: List[str] = []
        self.most_frequent_words: Set[str] = set()
        self.least_frequent_words: Set[str] = set()
        self.is_fitted = False

    @staticmethod
    def _is_post_review_leakage_feature(feature_name: str) -> bool:
        lname = feature_name.lower()
        return any(tok in lname for tok in _POST_REVIEW_LEAKAGE_TOKENS)

    def _artifact_paths(self, split_out_dir: str) -> Dict[str, str]:
        max_vocab = self.config.feature.max_vocab_token
        encoder = self.config.feature.encoder_token
        hand = self.config.feature.hand_token
        return {
            "labels": os.path.join(split_out_dir, f"labels_{max_vocab}_{encoder}_{hand}.tsv"),
            "ids": os.path.join(split_out_dir, f"ids_{max_vocab}_{encoder}_{hand}.tsv"),
            "svmlite": os.path.join(split_out_dir, f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt"),
        }

    def _feature_map_path(self) -> str:
        max_vocab = self.config.feature.max_vocab_token
        encoder = self.config.feature.encoder_token
        hand = self.config.feature.hand_token
        train_out_dir = os.path.join(
            self.config.data.combined_dir,
            "train",
            self.config.data.output_subdir,
        )
        return os.path.join(train_out_dir, f"features_{max_vocab}_{encoder}_{hand}.dat")

    def _load_split_papers(self, split: str) -> List[Any]:
        papers: List[Any] = []
        data_cfg = self.config.data
        print(f"Loading split={split} from {len(data_cfg.datasets)} datasets...")
        for idx, dataset_name in enumerate(data_cfg.datasets, start=1):
            p_dir = os.path.join(data_cfg.base_dir, dataset_name, split, data_cfg.reviews_subdir)
            s_dir = os.path.join(data_cfg.base_dir, dataset_name, split, data_cfg.parsed_subdir)

            if not (os.path.isdir(p_dir) and os.path.isdir(s_dir)):
                continue

            raw_dataset_papers = load_papers_from_dir(p_dir, s_dir)
            accepted_titles = self.label_resolver.load_accepted_titles(os.path.join(data_cfg.base_dir, dataset_name))
            dataset_papers, label_src_counts, skipped_unlabeled = self.label_resolver.label_dataset(
                raw_dataset_papers=raw_dataset_papers,
                accepted_titles=accepted_titles,
            )
            papers.extend(dataset_papers)
            print(
                f"  Dataset {idx}/{len(data_cfg.datasets)} {dataset_name}: "
                f"{len(dataset_papers)}/{len(raw_dataset_papers)} papers | "
                f"label_source={label_src_counts} | skipped_unlabeled={skipped_unlabeled}"
            )

        rng = random.Random(self.config.feature.shuffle_seed)
        rng.shuffle(papers)
        print(f"Loaded {len(papers)} labeled papers for split={split}.")
        return papers

    def _compute_corpus_stats(self, papers: Sequence[Any], cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        corpus_path = os.path.join(cache_dir, "corpus.pkl")

        if self.config.feature.preserve_corpus_cache and os.path.exists(corpus_path):
            corpus_words = pkl.load(open(corpus_path, "rb"))
        else:
            corpus_words: List[str] = []
            for paper in papers:
                content = paper.SCIENCEPARSE.get_paper_content()
                norm = normalize_text(
                    content,
                    only_char=self.config.preprocess.only_char,
                    lower=self.config.preprocess.lower,
                    stop_remove=self.config.preprocess.stop_remove,
                )
                corpus_words.extend(norm.split(" "))
            pkl.dump(corpus_words, open(corpus_path, "wb"))

        self.hfws, self.most_frequent_words, self.least_frequent_words = count_words(
            corpus_words,
            self.config.preprocess.hfw_proportion,
            self.config.preprocess.freq_proportion,
            self.config.preprocess.min_freq_threshold,
        )

    def fit(self, papers: Sequence[Any]) -> "FeaturePipeline":
        train_cache_dir = os.path.join(
            self.config.data.combined_dir,
            "train",
            self.config.data.output_subdir,
        )
        self._compute_corpus_stats(papers, train_cache_dir)

        self.id_to_feature = {}
        self._vectorize_papers(papers, is_train=True)
        self.is_fitted = True
        return self

    def _vectorize_papers(self, papers: Sequence[Any], is_train: bool) -> Tuple[List[Dict[int, float]], List[int], List[str], Set[str]]:
        if not self.config.feature.use_hand_features:
            return ([{} for _ in papers], [int(p.get_accepted() is True) for p in papers], [p.get_title() for p in papers], set())

        rows: List[Dict[int, float]] = []
        labels: List[int] = []
        titles: List[str] = []
        dropped_features: Set[str] = set()

        for paper in papers:
            labels.append(int(paper.get_accepted() is True))
            titles.append(paper.get_title())
            hand_features = extract_hand_features(
                paper,
                paper.SCIENCEPARSE,
                self.hfws,
                self.most_frequent_words,
                self.least_frequent_words,
            )

            row: Dict[int, float] = {}
            for feature_name, value in hand_features.items():
                if self.config.feature.drop_post_review_leakage and self._is_post_review_leakage_feature(feature_name):
                    dropped_features.add(feature_name)
                    continue

                if is_train and feature_name not in self.id_to_feature:
                    self.id_to_feature[feature_name] = len(self.id_to_feature)

                fid = self.id_to_feature.get(feature_name)
                if fid is not None and value != 0:
                    row[fid] = value
            rows.append(row)

        return rows, labels, titles, dropped_features

    def transform(self, papers: Sequence[Any]) -> Tuple[List[Dict[int, float]], List[int], List[str], Set[str]]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() or run_feature_extraction() first.")
        return self._vectorize_papers(papers, is_train=False)

    def fit_transform(self, papers: Sequence[Any]) -> Tuple[List[Dict[int, float]], List[int], List[str], Set[str]]:
        self.fit(papers)
        return self._vectorize_papers(papers, is_train=True)

    def _write_artifacts(self, split_out_dir: str, rows: Sequence[Dict[int, float]], labels: Sequence[int], titles: Sequence[str]) -> Dict[str, str]:
        os.makedirs(split_out_dir, exist_ok=True)
        paths = self._artifact_paths(split_out_dir)
        with open(paths["labels"], "w", encoding="utf-8") as out_labels, open(
            paths["ids"], "w", encoding="utf-8"
        ) as out_ids, open(paths["svmlite"], "w", encoding="utf-8") as out_svm:
            for idx, (row, label, title) in enumerate(zip(rows, labels, titles), start=1):
                out_labels.write(f"{int(label)}\n")
                out_ids.write(f"{idx}\t{title}\n")
                write_svmlite_row(int(label), row, out_svm)

        with open(paths["labels"], "r", encoding="utf-8") as lf:
            n_labels = sum(1 for _ in lf)
        if n_labels != len(rows):
            raise RuntimeError(
                f"Label count mismatch: expected {len(rows)} rows but found {n_labels} in {paths['labels']}."
            )

        return paths

    def run_feature_extraction(self) -> Dict[str, Dict[str, str]]:
        if "train" not in self.config.data.splits:
            raise ValueError("DataConfig.splits must include 'train' for fit/transform workflow.")

        for split in self.config.data.splits:
            os.makedirs(
                os.path.join(self.config.data.combined_dir, split, self.config.data.output_subdir),
                exist_ok=True,
            )

        artifacts: Dict[str, Dict[str, str]] = {}

        print("Extracting train split (fit + transform)...")
        train_papers = self._load_split_papers("train")
        train_rows, train_labels, train_titles, dropped = self.fit_transform(train_papers)
        if dropped:
            print(f"Dropped leakage-like features on train: {sorted(dropped)}")
        train_out_dir = os.path.join(self.config.data.combined_dir, "train", self.config.data.output_subdir)
        artifacts["train"] = self._write_artifacts(train_out_dir, train_rows, train_labels, train_titles)

        feature_map_path = self._feature_map_path()
        save_features_to_file(self.id_to_feature, feature_map_path)

        for split in self.config.data.splits:
            if split == "train":
                continue
            print(f"Extracting {split} split (transform only)...")
            split_papers = self._load_split_papers(split)
            rows, labels, titles, dropped = self.transform(split_papers)
            if dropped:
                print(f"Dropped leakage-like features on {split}: {sorted(dropped)}")
            split_out_dir = os.path.join(self.config.data.combined_dir, split, self.config.data.output_subdir)
            artifacts[split] = self._write_artifacts(split_out_dir, rows, labels, titles)

        return artifacts

    @staticmethod
    def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }

    @staticmethod
    def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
        print(
            f"{prefix} Acc: {metrics['accuracy']*100:.2f}% | "
            f"BalAcc: {metrics['balanced_accuracy']*100:.2f}% | "
            f"F1: {metrics['f1']*100:.2f}% | "
            f"Prec: {metrics['precision']*100:.2f}% | "
            f"Rec: {metrics['recall']*100:.2f}%"
        )

    def _pick_threshold(self, y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
        if self.config.training.use_probability_midpoints:
            uniq = np.unique(np.asarray(proba, dtype=float))
            if uniq.size == 0:
                thresholds = np.array([0.5], dtype=float)
            elif uniq.size == 1:
                thresholds = np.array([max(0.0, uniq[0] - 1e-6), min(1.0, uniq[0] + 1e-6)], dtype=float)
            else:
                mids = (uniq[:-1] + uniq[1:]) / 2.0
                thresholds = np.concatenate(([0.0], mids, [1.0]))
        else:
            thresholds = np.linspace(
                self.config.training.threshold_min,
                self.config.training.threshold_max,
                self.config.training.threshold_steps,
            )

        best_t = 0.5
        best_f1 = -1.0
        for t in thresholds:
            preds = (proba >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t, best_f1

    @staticmethod
    def _standardize(X_train: np.ndarray, X_dev: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return (X_train - mean) / std, (X_dev - mean) / std, (X_test - mean) / std

    def _load_training_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        data_dir = self.config.training.data_dir or self.config.data.combined_dir
        max_vocab = self.config.feature.max_vocab_token
        encoder = self.config.feature.encoder_token
        hand = self.config.feature.hand_token

        train_file = os.path.join(data_dir, "train", self.config.data.output_subdir, f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt")
        dev_file = os.path.join(data_dir, "dev", self.config.data.output_subdir, f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt")
        test_file = os.path.join(data_dir, "test", self.config.data.output_subdir, f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt")

        print(f"Loading training data from {train_file}...")
        X_train_sp, y_train = load_svmlight_file(train_file)
        n_features = X_train_sp.shape[1]

        print("Loading dev/test data...")
        X_dev_sp, y_dev = load_svmlight_file(dev_file, n_features=n_features)
        X_test_sp, y_test = load_svmlight_file(test_file, n_features=n_features)

        X_train = X_train_sp.toarray()
        X_dev = X_dev_sp.toarray()
        X_test = X_test_sp.toarray()

        return (X_train, y_train.astype(int)), (X_dev, y_dev.astype(int)), (X_test, y_test.astype(int))

    def _instantiate_model(self, spec: Dict[str, Any]) -> Any:
        model_type = spec.get("model_type")
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type={model_type!r}. Supported: {sorted(_MODEL_REGISTRY)}")
        model_cls = _MODEL_REGISTRY[model_type]
        hyperparams = spec.get("hyperparams", {})
        return model_cls(**hyperparams)

    def _train_eval_candidate(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        model.fit(X_train, y_train)
        dev_proba = model.predict_proba(X_dev)[:, 1]
        best_t, dev_best_f1 = self._pick_threshold(y_dev, dev_proba)

        dev_pred = (dev_proba >= best_t).astype(int)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= best_t).astype(int)

        result = {
            "name": name,
            "model": model,
            "threshold": best_t,
            "dev_best_f1": dev_best_f1,
            "dev": self.metrics_dict(y_dev, dev_pred),
            "test": self.metrics_dict(y_test, test_pred),
        }
        return result, dev_proba, test_proba

    def run_training(self) -> Dict[str, Any]:
        if not self.config.training.enabled:
            print("Training disabled by config (TrainingConfig.enabled=False).")
            return {}

        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = self._load_training_data()
        print(f"Dataset loaded. Train shape={X_train.shape}, Dev shape={X_dev.shape}, Test shape={X_test.shape}")

        majority_class = int(np.bincount(y_train).argmax())
        maj_dev = np.full_like(y_dev, majority_class)
        maj_test = np.full_like(y_test, majority_class)
        print("Majority baseline:")
        self.print_metrics("  Dev", self.metrics_dict(y_dev, maj_dev))
        self.print_metrics("  Test", self.metrics_dict(y_test, maj_test))
        print("-" * 50)

        X_train_scaled, X_dev_scaled, X_test_scaled = self._standardize(X_train, X_dev, X_test)

        results: List[Dict[str, Any]] = []
        model_dev_proba: Dict[str, np.ndarray] = {}
        model_test_proba: Dict[str, np.ndarray] = {}

        for spec in self.config.training.model_candidates:
            name = spec.get("name", spec.get("model_type", "unnamed"))
            requires_scaling = bool(spec.get("requires_scaling", True))
            model = self._instantiate_model(spec)
            Xt, Xd, Xte = (X_train_scaled, X_dev_scaled, X_test_scaled) if requires_scaling else (X_train, X_dev, X_test)

            print(f"Training {name}...")
            result, dev_proba, test_proba = self._train_eval_candidate(name, model, Xt, y_train, Xd, y_dev, Xte, y_test)
            results.append(result)
            model_dev_proba[name] = dev_proba
            model_test_proba[name] = test_proba
            print(f"  Threshold tuned on dev: {result['threshold']:.4f}")
            self.print_metrics("  Dev", result["dev"])
            self.print_metrics("  Test", result["test"])
            print("-" * 50)

        if not results:
            raise RuntimeError("No model finished successfully.")

        ranked = sorted(results, key=lambda r: (r["dev"]["f1"], r["dev"]["balanced_accuracy"]), reverse=True)
        best = ranked[0]
        print("Best single model by dev metrics:")
        print(f"  {best['name']} | threshold={best['threshold']:.4f}")
        self.print_metrics("  Dev", best["dev"])
        self.print_metrics("  Test", best["test"])
        print("-" * 50)

        top_k = max(0, int(self.config.training.ensemble_top_k))
        blend_result: Optional[Dict[str, Any]] = None
        if top_k >= 2 and len(ranked) >= top_k:
            top_names = [r["name"] for r in ranked[:top_k]]
            blend_dev = np.mean([model_dev_proba[n] for n in top_names], axis=0)
            blend_test = np.mean([model_test_proba[n] for n in top_names], axis=0)
            blend_t, _ = self._pick_threshold(y_dev, blend_dev)
            blend_dev_pred = (blend_dev >= blend_t).astype(int)
            blend_test_pred = (blend_test >= blend_t).astype(int)
            print(f"Top-{top_k} probability blend: {top_names}")
            print(f"  Threshold tuned on dev: {blend_t:.4f}")
            blend_dev_metrics = self.metrics_dict(y_dev, blend_dev_pred)
            blend_test_metrics = self.metrics_dict(y_test, blend_test_pred)
            self.print_metrics("  Dev", blend_dev_metrics)
            self.print_metrics("  Test", blend_test_metrics)
            print("-" * 50)
            blend_result = {
                "names": top_names,
                "threshold": blend_t,
                "dev": blend_dev_metrics,
                "test": blend_test_metrics,
            }

        return {
            "best": best,
            "ranked": ranked,
            "blend": blend_result,
        }

    def load_feature_map(self) -> None:
        feature_map_path = self._feature_map_path()
        self.id_to_feature = read_features(feature_map_path)
        self.is_fitted = True

    def run(self) -> Dict[str, Any]:
        extraction_artifacts = self.run_feature_extraction()
        training_summary = self.run_training() if self.config.training.enabled else {}
        return {
            "artifacts": extraction_artifacts,
            "training": training_summary,
        }
