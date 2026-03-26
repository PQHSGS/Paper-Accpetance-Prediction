from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .base import BaseBinaryClassifier


@dataclass
class _TreeNode:
    is_leaf: bool
    proba1: float

    feature_index: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None


class DecisionTreeClassifier(BaseBinaryClassifier):
    """
    Weighted CART-style decision tree for binary classification.

    Supports:
    - criterion: 'gini' or 'entropy'
    - max_depth
    - min_samples_split
    - min_samples_leaf
    - sample_weight (handled by BaseBinaryClassifier and passed through)

    Note: This implementation is optimized for correctness and clarity, not for
    very large datasets.
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        # Convenience alias sometimes used in other implementations.
        min_node: Optional[int] = None,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(balance=balance)
        self.criterion = str(criterion).lower()
        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be one of {'gini','entropy'}")

        self.max_depth = max_depth
        self.min_samples_split = int(min_node if min_node is not None else min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self._root: Optional[_TreeNode] = None
        self._n_features: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

    # --------------------
    # Internal fitting
    # --------------------
    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        self._n_features = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._root = self._build_tree(X, y_bin, sample_weight, depth=0)

    def _impurity(self, p1: float) -> float:
        # p1 is weighted P(y=1)
        p1 = float(p1)
        p0 = 1.0 - p1
        if self.criterion == "gini":
            return 2.0 * p1 * p0
        # entropy
        eps = 1e-12
        p1 = np.clip(p1, eps, 1.0 - eps)
        return -p1 * np.log(p1) - (1.0 - p1) * np.log(1.0 - p1)

    def _node_proba1(self, y_bin: np.ndarray, sample_weight: np.ndarray) -> float:
        total = float(sample_weight.sum())
        if total <= 0:
            return 0.5
        w1 = float(sample_weight[y_bin == 1].sum())
        return w1 / total

    def _stopping_condition(self, y_bin: np.ndarray, sample_weight: np.ndarray, depth: int) -> bool:
        n = y_bin.shape[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n < self.min_samples_split:
            return True
        if np.all(y_bin == 0) or np.all(y_bin == 1):
            return True
        if float(sample_weight.sum()) <= 0:
            return True
        return False

    def _choose_feature_indices(self) -> np.ndarray:
        assert self._n_features is not None
        m = self.max_features
        if m is None:
            return np.arange(self._n_features, dtype=int)
        if isinstance(m, str):
            m_lower = m.lower()
            if m_lower == "sqrt":
                k = max(1, int(np.sqrt(self._n_features)))
            elif m_lower in {"log2", "ln"}:
                k = max(1, int(np.log2(max(2, self._n_features))))
            else:
                raise ValueError("max_features string must be one of {None,'sqrt','log2'}")
        elif isinstance(m, float):
            if m <= 0 or m > 1:
                raise ValueError("If max_features is float, it must be in (0, 1]")
            k = max(1, int(np.ceil(m * self._n_features)))
        else:
            k = int(m)
            k = max(1, min(self._n_features, k))
        assert self._rng is not None
        return self._rng.choice(self._n_features, size=k, replace=False).astype(int)

    def _build_tree(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray, depth: int) -> _TreeNode:
        p1 = self._node_proba1(y_bin, sample_weight)
        if self._stopping_condition(y_bin, sample_weight, depth):
            return _TreeNode(is_leaf=True, proba1=p1)

        parent_imp = self._impurity(p1)

        n_samples, n_features = X.shape
        feature_indices = self._choose_feature_indices()

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        min_leaf = self.min_samples_leaf
        if min_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        for j in feature_indices:
            col = X[:, j]
            order = np.argsort(col, kind="mergesort")
            col_sorted = col[order]
            y_sorted = y_bin[order]
            sw_sorted = sample_weight[order]

            # Candidate split positions are between distinct values.
            distinct = np.nonzero(col_sorted[1:] != col_sorted[:-1])[0]
            if distinct.size == 0:
                continue

            prefix_total = np.cumsum(sw_sorted)
            prefix_w1 = np.cumsum(sw_sorted * y_sorted)
            total_weight = float(prefix_total[-1])
            if total_weight <= 0:
                continue

            for i in distinct:
                left_count = i + 1
                right_count = n_samples - left_count
                if left_count < min_leaf or right_count < min_leaf:
                    continue

                left_total = float(prefix_total[i])
                right_total = total_weight - left_total
                if left_total <= 0 or right_total <= 0:
                    continue

                left_p1 = float(prefix_w1[i]) / left_total
                right_p1 = float(prefix_w1[-1] - prefix_w1[i]) / right_total

                left_imp = self._impurity(left_p1)
                right_imp = self._impurity(right_p1)
                child_imp = (left_total / total_weight) * left_imp + (right_total / total_weight) * right_imp
                gain = parent_imp - child_imp

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(j)
                    best_threshold = float((col_sorted[i] + col_sorted[i + 1]) / 2.0)
                    best_left_idx = order[: left_count]
                    best_right_idx = order[left_count:]

        # If no split improves impurity, make leaf.
        if best_feature is None or best_left_idx is None or best_right_idx is None:
            return _TreeNode(is_leaf=True, proba1=p1)
        if best_gain <= 1e-12:
            return _TreeNode(is_leaf=True, proba1=p1)

        left_node = self._build_tree(X[best_left_idx], y_bin[best_left_idx], sample_weight[best_left_idx], depth + 1)
        right_node = self._build_tree(
            X[best_right_idx], y_bin[best_right_idx], sample_weight[best_right_idx], depth + 1
        )
        return _TreeNode(
            is_leaf=False,
            proba1=p1,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("Internal error: tree root is not set after fit().")

        n = X.shape[0]
        out = np.empty(n, dtype=float)

        for i in range(n):
            node = self._root
            while not node.is_leaf:
                assert node.feature_index is not None
                assert node.threshold is not None
                if X[i, node.feature_index] <= node.threshold:
                    assert node.left is not None
                    node = node.left
                else:
                    assert node.right is not None
                    node = node.right
            out[i] = node.proba1
        return out

