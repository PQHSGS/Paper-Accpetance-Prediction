import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from .config.pipeline import PipelineConfig
from .feature import get_feature_methods, read_features, save_features_to_file, write_svmlite_row
from .entities import FeaturedData, ProcessData
from .preprocess import get_data_preprocess_methods, get_feature_preprocess_methods
from .preprocess.parsing import load_papers_from_dir
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

MatrixTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]
SplitMatrices = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class _LabelResolver:
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
            value = accepted.strip().lower()
            if value in {"true", "1", "accept", "accepted", "yes"}:
                return True
            if value in {"false", "0", "reject", "rejected", "no"}:
                return False
        return None

    @staticmethod
    def _get_avg_recommendation(paper: Any) -> Optional[float]:
        recommendations: List[float] = []
        for review in paper.get_reviews():
            recommendation = review.get_recommendation()
            try:
                if recommendation is not None and str(recommendation).strip() != "":
                    recommendations.append(float(recommendation))
            except Exception:
                continue
        if not recommendations:
            return None
        return float(sum(recommendations) / len(recommendations))

    @staticmethod
    def load_accepted_titles(dataset_root: str) -> Set[str]:
        accepted_txt_path = os.path.join(dataset_root, "acl_accepted.txt")
        accepted_titles: Set[str] = set()
        if os.path.exists(accepted_txt_path):
            with open(accepted_txt_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    title = line.strip().lower()
                    if title:
                        accepted_titles.add(title)
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


class FeaturePipeline:
    """Stage-oriented pipeline.

    The intended flow is:
    load_raw_data -> preprocess_data -> extract_features -> load_features -> predict -> eval
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.id_to_feature: Dict[str, int] = {}
        self.is_feature_fitted = False
        self.model = self._instantiate_model(
            self.config.training.model, dict(self.config.training.model_param)
        )
        self.model_name = self.config.training.model

    def run(self, extract_features: bool = True) -> Dict[str, Any]:
        extraction_artifacts = self.run_feature_extraction() if extract_features else {}
        training_summary: Dict[str, Any] = {}

        # Keep the high-level flow explicit:
        # load raw_data -> preprocessed_data -> extract feature -> load feature -> predict -> eval
        if self.config.training.enabled:
            (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = self.prepare_input()
            baseline = self.majority_baseline(y_train, y_dev, y_test)

            print(f"Training {self.config.training.model} with params={self.config.training.model_param}...")
            self.fit(self.model, X_train, y_train)
            result = self.eval(self.config.training.model, self.model, X_dev, y_dev, X_test, y_test)
            print(f"  Decision threshold: {result['decision_threshold']:.2f}")
            self.print_metrics("  Dev", result["dev"])
            self.print_metrics("  Test", result["test"])
            print("-" * 50)

            training_summary = {
                "model": self.config.training.model,
                "model_param": dict(self.config.training.model_param),
                "majority_baseline": baseline,
                "result": result,
            }

        return {
            "artifacts": extraction_artifacts,
            "training": training_summary,
        }

    def run_feature_extraction(self) -> Dict[str, Dict[str, str]]:
        if "train" not in self.config.data.splits:
            raise ValueError("DataConfig.splits must include 'train' for fit/transform workflow.")

        for split in self.config.data.splits:
            os.makedirs(
                os.path.join(self.config.data.combined_dir, split, self.config.data.output_subdir),
                exist_ok=True,
            )

        artifacts: Dict[str, Dict[str, str]] = {}
        for split in self.config.data.splits:
            fit_mode = split == "train"
            phase = "fit + transform" if fit_mode else "transform only"
            print(f"Extracting {split} split ({phase})...")
            raw_data = self.load_raw_data(split)
            preprocessed_data = self.preprocess_data(raw_data)
            featured = self.extract_features(preprocessed_data, fit_mode=fit_mode)
            featured = self._write_artifacts(featured)
            if featured.dropped_features:
                print(f"Dropped leakage-like features on {split}: {sorted(featured.dropped_features)}")
            artifacts[split] = featured.artifact_paths
            if fit_mode:
                save_features_to_file(self.id_to_feature, self._feature_map_path())

        return artifacts

    def load_raw_data(self, split: str) -> ProcessData:
        papers: List[Any] = []
        label_source_counts = {
            "json.accepted": 0,
            "acl_accepted.txt": 0,
            "review.recommendation>=threshold": 0,
        }
        skipped_unlabeled = 0

        resolver = _LabelResolver(
            allow_recommendation_fallback=self.config.preprocess.allow_recommendation_fallback,
            recommendation_threshold=self.config.preprocess.recommendation_threshold,
        )

        data_cfg = self.config.data
        print(f"Loading split={split} from {len(data_cfg.datasets)} datasets...")
        for idx, dataset_name in enumerate(data_cfg.datasets, start=1):
            reviews_dir = os.path.join(data_cfg.base_dir, dataset_name, split, data_cfg.reviews_subdir)
            parsed_dir = os.path.join(data_cfg.base_dir, dataset_name, split, data_cfg.parsed_subdir)

            if not (os.path.isdir(reviews_dir) and os.path.isdir(parsed_dir)):
                continue

            raw_dataset_papers = load_papers_from_dir(reviews_dir, parsed_dir)
            accepted_titles = resolver.load_accepted_titles(os.path.join(data_cfg.base_dir, dataset_name))

            dataset_papers: List[Any] = []
            dataset_label_src_counts = {
                "json.accepted": 0,
                "acl_accepted.txt": 0,
                "review.recommendation>=threshold": 0,
            }
            dataset_skipped = 0
            for paper in raw_dataset_papers:
                try:
                    label, src = resolver.resolve_label(paper, accepted_titles)
                except ValueError:
                    dataset_skipped += 1
                    continue

                paper.ACCEPTED = bool(label)
                dataset_label_src_counts[src] += 1
                dataset_papers.append(paper)

            papers.extend(dataset_papers)
            skipped_unlabeled += dataset_skipped
            for key, value in dataset_label_src_counts.items():
                label_source_counts[key] += value

            print(
                f"  Dataset {idx}/{len(data_cfg.datasets)} {dataset_name}: "
                f"{len(dataset_papers)}/{len(raw_dataset_papers)} papers | "
                f"label_source={dataset_label_src_counts} | skipped_unlabeled={dataset_skipped}"
            )

        rng = np.random.default_rng(self.config.preprocess.shuffle_seed)
        if papers:
            order = rng.permutation(len(papers))
            papers = [papers[i] for i in order]

        labels = [int(paper.get_accepted() is True) for paper in papers]
        titles = [paper.get_title() for paper in papers]

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

    def preprocess_data(self, data: ProcessData) -> ProcessData:
        cache_dir = os.path.join(
            self.config.data.combined_dir,
            "train",
            self.config.data.output_subdir,
        )
        methods = get_data_preprocess_methods(self.config.preprocess.methods)
        context = {"cache_dir": cache_dir}

        current = data
        for method in methods:
            current = method(current, self.config.preprocess, self.config.feature, context)
        return current

    def extract_features(self, data: ProcessData, fit_mode: bool) -> FeaturedData:
        methods = get_feature_methods(self.config.feature.methods)
        if fit_mode:
            self.id_to_feature = {}
        elif not self.is_feature_fitted:
            raise RuntimeError("FeaturePipeline is not fitted. Run train split first or call load_feature_map().")

        rows: List[Dict[int, float]] = []
        dropped_features: Set[str] = set()

        if not self.config.feature.use_hand_features:
            rows = [{} for _ in data.papers]
        else:
            for paper in data.papers:
                merged_features: Dict[str, float] = {}
                for method in methods:
                    method_features = method(paper, data, self.config.feature)
                    merged_features.update(method_features)

                row: Dict[int, float] = {}
                for feature_name, value in merged_features.items():
                    if self.config.feature.drop_post_review_leakage and self._is_post_review_leakage_feature(feature_name):
                        dropped_features.add(feature_name)
                        continue

                    if fit_mode and feature_name not in self.id_to_feature:
                        self.id_to_feature[feature_name] = len(self.id_to_feature)

                    fid = self.id_to_feature.get(feature_name)
                    if fid is not None and value != 0:
                        row[fid] = value
                rows.append(row)

        if fit_mode:
            self.is_feature_fitted = True

        return FeaturedData(
            split=data.split,
            rows=rows,
            labels=list(data.labels),
            titles=list(data.titles),
            id_to_feature=dict(self.id_to_feature),
            dropped_features=dropped_features,
            metadata=dict(data.metadata),
        )


    def _write_artifacts(self, featured: FeaturedData) -> FeaturedData:
        split_out_dir = os.path.join(self.config.data.combined_dir, featured.split, self.config.data.output_subdir)
        os.makedirs(split_out_dir, exist_ok=True)

        paths = self._artifact_paths(split_out_dir)
        with open(paths["labels"], "w", encoding="utf-8") as out_labels, open(
            paths["ids"], "w", encoding="utf-8"
        ) as out_ids, open(paths["svmlite"], "w", encoding="utf-8") as out_svm:
            for idx, (row, label, title) in enumerate(zip(featured.rows, featured.labels, featured.titles), start=1):
                out_labels.write(f"{int(label)}\n")
                out_ids.write(f"{idx}\t{title}\n")
                write_svmlite_row(int(label), row, out_svm)

        with open(paths["labels"], "r", encoding="utf-8") as lf:
            n_labels = sum(1 for _ in lf)
        if n_labels != len(featured.rows):
            raise RuntimeError(
                f"Label count mismatch: expected {len(featured.rows)} rows but found {n_labels} in {paths['labels']}."
            )

        featured.artifact_paths = paths
        return featured

    def load_features(self) -> SplitMatrices:
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

    def prepare_input(self) -> SplitMatrices:
        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) =  self.load_features()
        print(f"Dataset loaded. Train shape={X_train.shape}, Dev shape={X_dev.shape}, Test shape={X_test.shape}")
        X_train, X_dev, X_test = self._preprocess_feature((X_train, X_dev, X_test))
        return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

    def majority_baseline(
        self,
        y_train: np.ndarray,
        y_dev: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        majority_class = int(np.bincount(y_train).argmax())
        maj_dev = np.full_like(y_dev, majority_class)
        maj_test = np.full_like(y_test, majority_class)
        baseline = {
            "class": majority_class,
            "dev": self.metrics_dict(y_dev, maj_dev),
            "test": self.metrics_dict(y_test, maj_test),
        }
        print("Majority baseline:")
        self.print_metrics("  Dev", baseline["dev"])
        self.print_metrics("  Test", baseline["test"])
        print("-" * 50)
        return baseline



    @staticmethod
    def fit(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    @staticmethod
    def predict_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (probabilities >= threshold).astype(int)

    def eval(
        self,
        name: str,
        model: Any,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        dev_proba = self.predict_proba(model, X_dev)
        dev_pred = self.predict_labels(dev_proba)
        test_proba = self.predict_proba(model, X_test)
        test_pred = self.predict_labels(test_proba)

        return {
            "name": name,
            "decision_threshold": 0.5,
            "dev": self.metrics_dict(y_dev, dev_pred),
            "test": self.metrics_dict(y_test, test_pred),
        }

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

    def _preprocess_feature(self, matrices: MatrixTriplet) -> MatrixTriplet:
        methods = get_feature_preprocess_methods(self.config.training.preprocess_methods)
        context: Dict[str, Any] = {}
        current = matrices
        for method in methods:
            current = method(current, self.config.training, context)
        return current

    def _instantiate_model(self, model_name: str, model_param: Dict[str, Any]) -> Any:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(f"Unsupported model={model_name!r}. Supported: {sorted(_MODEL_REGISTRY)}")
        model_cls = _MODEL_REGISTRY[model_name]
        return model_cls(**model_param)

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

    def _artifact_paths(self, split_out_dir: str) -> Dict[str, str]:
        max_vocab = self.config.feature.max_vocab_token
        encoder = self.config.feature.encoder_token
        hand = self.config.feature.hand_token
        return {
            "labels": os.path.join(split_out_dir, f"labels_{max_vocab}_{encoder}_{hand}.tsv"),
            "ids": os.path.join(split_out_dir, f"ids_{max_vocab}_{encoder}_{hand}.tsv"),
            "svmlite": os.path.join(split_out_dir, f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt"),
        }

    @staticmethod
    def _is_post_review_leakage_feature(feature_name: str) -> bool:
        lname = feature_name.lower()
        return any(tok in lname for tok in _POST_REVIEW_LEAKAGE_TOKENS)

    def load_feature_map(self) -> None:
        feature_map_path = self._feature_map_path()
        self.id_to_feature = read_features(feature_map_path)
        self.is_feature_fitted = True
