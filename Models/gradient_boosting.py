from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseBinaryClassifier


@dataclass
class _RegTreeNode:
    is_leaf: bool
    value: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_RegTreeNode"] = None
    right: Optional["_RegTreeNode"] = None


class _WeightedRegressionTree:
    """
    Regression tree that fits continuous targets with per-sample weights,
    minimizing weighted squared error with a simple L2 regularization on leaf values.
    """

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        l2_leaf_reg: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.random_state = random_state
        self._root: Optional[_RegTreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        self._root = self._build_tree(X, y, w, depth=0)
        return self

    def _leaf_value(self, y: np.ndarray, w: np.ndarray) -> float:
        A = float(w.sum())
        if A <= 0:
            return 0.0
        B = float((w * y).sum())
        return B / (A + self.l2_leaf_reg)

    def _node_loss(self, y: np.ndarray, w: np.ndarray) -> float:
        """
        Weighted squared error for constant leaf with L2 leaf regularization.
        """
        A = float(w.sum())
        if A <= 0:
            return 0.0
        B = float((w * y).sum())
        C2 = float((w * (y * y)).sum())
        # c = B / (A + lam)
        denom = A + self.l2_leaf_reg
        c = B / denom
        # loss = Σ w*(y - c)^2
        return C2 - 2.0 * c * B + (c * c) * A

    def _stopping(self, y: np.ndarray, w: np.ndarray, depth: int) -> bool:
        n = y.shape[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n < self.min_samples_split:
            return True
        if float(w.sum()) <= 0:
            return True
        # If all y are the same, no split helps.
        if np.all(y == y[0]):
            return True
        return False

    def _build_tree(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> _RegTreeNode:
        value = self._leaf_value(y, w)
        if self._stopping(y, w, depth):
            return _RegTreeNode(is_leaf=True, value=value)

        parent_loss = self._node_loss(y, w)
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        min_leaf = self.min_samples_leaf
        if min_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        # Consider all features (simpler; RF could add feature subsampling later if desired).
        for j in range(n_features):
            col = X[:, j]
            order = np.argsort(col, kind="mergesort")
            col_sorted = col[order]
            y_sorted = y[order]
            w_sorted = w[order]

            distinct = np.nonzero(col_sorted[1:] != col_sorted[:-1])[0]
            if distinct.size == 0:
                continue

            prefix_w = np.cumsum(w_sorted)
            prefix_wy = np.cumsum(w_sorted * y_sorted)
            prefix_wy2 = np.cumsum(w_sorted * (y_sorted * y_sorted))
            total_w = float(prefix_w[-1])
            if total_w <= 0:
                continue

            total_loss = parent_loss

            for i in distinct:
                left_count = i + 1
                right_count = n_samples - left_count
                if left_count < min_leaf or right_count < min_leaf:
                    continue

                A_l = float(prefix_w[i])
                A_r = total_w - A_l
                if A_l <= 0 or A_r <= 0:
                    continue

                B_l = float(prefix_wy[i])
                B_r = float(prefix_wy[-1] - prefix_wy[i])

                S2_l = float(prefix_wy2[i])
                S2_r = float(prefix_wy2[-1] - prefix_wy2[i])

                # child losses with regularized leaf constants
                denom_l = A_l + self.l2_leaf_reg
                denom_r = A_r + self.l2_leaf_reg
                c_l = B_l / denom_l
                c_r = B_r / denom_r
                loss_l = S2_l - 2.0 * c_l * B_l + (c_l * c_l) * A_l
                loss_r = S2_r - 2.0 * c_r * B_r + (c_r * c_r) * A_r
                split_loss = loss_l + loss_r

                gain = total_loss - split_loss
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(j)
                    best_threshold = float((col_sorted[i] + col_sorted[i + 1]) / 2.0)
                    best_left_idx = order[:left_count]
                    best_right_idx = order[left_count:]

        if best_feature is None or best_left_idx is None or best_right_idx is None:
            return _RegTreeNode(is_leaf=True, value=value)
        if best_gain <= 1e-12:
            return _RegTreeNode(is_leaf=True, value=value)

        left_node = self._build_tree(
            X[best_left_idx], y[best_left_idx], w[best_left_idx], depth=depth + 1
        )
        right_node = self._build_tree(
            X[best_right_idx], y[best_right_idx], w[best_right_idx], depth=depth + 1
        )
        return _RegTreeNode(
            is_leaf=False,
            value=value,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("Regression tree not fitted.")
        X = np.asarray(X, dtype=float)
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
            out[i] = node.value
        return out


class GradientBoostingClassifier(BaseBinaryClassifier):
    """
    Binary gradient boosting for logistic loss using additive trees on log-odds.

    - Maintains an additive model F(x) on the log-odds scale:
        p(x) = sigmoid(F(x))
    - At each iteration, fits a weighted regression tree to the negative gradient /
      pseudo-residuals under a logistic loss approximation.
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        l2_leaf_reg: float = 0.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(balance=balance)
        self.n_estimators = int(n_estimators)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.random_state = random_state

        self.init_log_odds_: float = 0.0
        self.trees_: list[_WeightedRegressionTree] = []

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        sw = sample_weight.astype(float, copy=False)
        total_w = float(sw.sum())
        if total_w <= 0:
            sw = np.ones_like(sw, dtype=float)
            total_w = float(sw.sum())

        # Initialize with weighted log-odds of class 1.
        p = float((sw * y_bin).sum() / total_w)
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        self.init_log_odds_ = float(np.log(p / (1.0 - p)))

        F = np.full(X.shape[0], self.init_log_odds_, dtype=float)
        self.trees_ = []

        for t in range(self.n_estimators):
            p_pred = self._sigmoid(F)
            # Pseudo-residual / negative gradient for logistic loss.
            residual = y_bin.astype(float) - p_pred
            # Weight for Newton step approximation (diagonal of Hessian).
            tree_weights = sw * p_pred * (1.0 - p_pred)
            if float(tree_weights.sum()) <= 0:
                tree_weights = sw.copy()

            tree = _WeightedRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                l2_leaf_reg=self.l2_leaf_reg,
                random_state=None if self.random_state is None else (self.random_state + t),
            )
            tree.fit(X, residual, tree_weights)
            update = tree.predict(X)
            F = F + self.learning_rate * update
            self.trees_.append(tree)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self.init_log_odds_, dtype=float)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
        return self._sigmoid(F)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return (self._predict_proba_internal(X) >= 0.5).astype(int)

