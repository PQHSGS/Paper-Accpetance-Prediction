import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from .config.pipeline import PipelineConfig
from .feature import get_feature_methods, read_features, save_features_to_file, write_svmlite_row
from .entities import FeaturedData, ProcessData
from .preprocess import get_data_preprocess_methods, get_feature_preprocess_methods
from .preprocess.loading import load_split_data
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


class FeaturePipeline:
    """Stage-oriented pipeline.

    The intended flow is:
    load_raw_data -> preprocess_data -> extract_features -> load_features -> train -> predict -> eval
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.id_to_feature: Dict[str, int] = {}
        self.is_feature_fitted = False

    def run(self) -> Dict[str, Any]:
        extraction_artifacts = self.run_feature_extraction()

        # Keep the high-level flow explicit:
        # load raw_data -> preprocessed_data -> extract feature -> load feature -> train -> predict -> eval
        loaded_features = self.load_features() if self.config.training.enabled else None
        training_summary = self.train(loaded_features) if self.config.training.enabled else {}
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
            featured = self._process_split(split=split, fit_mode=fit_mode)
            if featured.dropped_features:
                print(f"Dropped leakage-like features on {split}: {sorted(featured.dropped_features)}")
            artifacts[split] = featured.artifact_paths
            if fit_mode:
                save_features_to_file(self.id_to_feature, self._feature_map_path())

        return artifacts
    
    def _process_split(self, split: str, fit_mode: bool) -> FeaturedData:
        raw_data = self.load_raw_data(split)
        preprocessed_data = self.preprocess_data(raw_data)
        featured = self.extract_features(preprocessed_data, fit_mode=fit_mode)
        return self._write_artifacts(featured)


    def load_raw_data(self, split: str) -> ProcessData:
        return load_split_data(self.config, split)

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

    def load_features(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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

    def train(
        self,
        loaded_features: Optional[
            Tuple[
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
            ]
        ] = None,
    ) -> Dict[str, Any]:
        if not self.config.training.enabled:
            print("Training disabled by config (TrainingConfig.enabled=False).")
            return {}

        if loaded_features is None:
            loaded_features = self.load_features()

        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = loaded_features
        print(f"Dataset loaded. Train shape={X_train.shape}, Dev shape={X_dev.shape}, Test shape={X_test.shape}")

        majority_class = int(np.bincount(y_train).argmax())
        maj_dev = np.full_like(y_dev, majority_class)
        maj_test = np.full_like(y_test, majority_class)
        print("Majority baseline:")
        self.print_metrics("  Dev", self.metrics_dict(y_dev, maj_dev))
        self.print_metrics("  Test", self.metrics_dict(y_test, maj_test))
        print("-" * 50)

        results: List[Dict[str, Any]] = []
        model_dev_proba: Dict[str, np.ndarray] = {}
        model_test_proba: Dict[str, np.ndarray] = {}
        scaled_matrices: Optional[MatrixTriplet] = None

        for spec in self.config.training.model_candidates:
            name = spec.get("name", spec.get("model_type", "unnamed"))
            requires_scaling = bool(spec.get("requires_scaling", True))
            model = self._instantiate_model(spec)
            if requires_scaling:
                if scaled_matrices is None:
                    scaled_matrices = self._preprocess_feature((X_train, X_dev, X_test))
                X_train, X_dev, X_test = scaled_matrices

            print(f"Training {name}...")
            model = self.fit(model, X_train, y_train)
            result, dev_proba, test_proba = self.eval(name, model, X_dev, y_dev, X_test, y_test)
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

    def fit(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    @staticmethod
    def predict_labels(probabilities: np.ndarray, threshold: float) -> np.ndarray:
        return (probabilities >= threshold).astype(int)

    @staticmethod
    def _score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return FeaturePipeline.metrics_dict(y_true, y_pred)

    def eval(
        self,
        name: str,
        model: Any,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        dev_proba = self.predict_proba(model, X_dev)
        best_t, dev_best_f1 = self._pick_threshold(y_dev, dev_proba)

        dev_pred = self.predict_labels(dev_proba, best_t)
        test_proba = self.predict_proba(model, X_test)
        test_pred = self.predict_labels(test_proba, best_t)

        result = {
            "name": name,
            "model": model,
            "threshold": best_t,
            "dev_best_f1": dev_best_f1,
            "dev": self._score(y_dev, dev_pred),
            "test": self._score(y_test, test_pred),
        }
        return result, dev_proba, test_proba

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

    def _preprocess_feature(self, matrices: MatrixTriplet) -> MatrixTriplet:
        methods = get_feature_preprocess_methods(self.config.training.preprocess_methods)
        context: Dict[str, Any] = {}
        current = matrices
        for method in methods:
            current = method(current, self.config.training, context)
        return current

    def _instantiate_model(self, spec: Dict[str, Any]) -> Any:
        model_type = spec.get("model_type")
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type={model_type!r}. Supported: {sorted(_MODEL_REGISTRY)}")
        model_cls = _MODEL_REGISTRY[model_type]
        hyperparams = spec.get("hyperparams", {})
        return model_cls(**hyperparams)

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

