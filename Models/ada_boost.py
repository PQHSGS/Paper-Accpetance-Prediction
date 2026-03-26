from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseBinaryClassifier
from .decision_tree import DecisionTreeClassifier


class AdaBoostClassifier(BaseBinaryClassifier):
    """
    AdaBoost.M1 for binary classification.

    Uses `DecisionTreeClassifier` as the base estimator, trained with `sample_weight`.

    Probability estimation:
    - Computes the ensemble margin F(x) = sum_t alpha_t * y_pred_sign
    - Converts margin to P(y=1|x) via sigmoid(2F)
      (This is a common monotonic mapping; it is not a perfect calibration method.)
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        base_max_depth: int = 1,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        super().__init__(balance=balance)
        self.n_estimators = int(n_estimators)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.learning_rate = float(learning_rate)
        self.criterion = criterion
        self.base_max_depth = int(base_max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.alphas_: list[float] = []

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        n = X.shape[0]
        w = sample_weight.astype(float, copy=False)
        total = float(w.sum())
        if total <= 0:
            w = np.ones(n, dtype=float)
            total = float(w.sum())
        w = w / total

        self.estimators_ = []
        self.alphas_ = []

        y_sign = 2.0 * y_bin.astype(float) - 1.0  # {-1, +1}

        for t in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                balance=False,
                criterion=self.criterion,
                max_depth=self.base_max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                random_state=None if self.random_state is None else (self.random_state + t),
            )
            tree.fit(X, y_bin, sample_weight=w)

            y_pred = tree.predict(X)
            y_pred01 = y_pred.astype(int)
            pred_sign = 2.0 * y_pred01.astype(float) - 1.0

            mis = (y_pred01 != y_bin).astype(float)
            err = float(np.sum(w * mis))

            # If perfectly classified, alpha -> infinity. Stop early.
            if err <= 1e-12:
                alpha = float("inf")
                self.estimators_.append(tree)
                self.alphas_.append(alpha)
                break

            # If error >= 0.5, the weak learner is worse than random.
            if err >= 0.5:
                # Stop instead of adding a negative/unstable alpha.
                break

            alpha = self.learning_rate * 0.5 * np.log((1.0 - err) / max(err, 1e-12))
            self.estimators_.append(tree)
            self.alphas_.append(float(alpha))

            # Update weights:
            # w_i <- w_i * exp(-alpha * y_i * pred_i)
            w = w * np.exp(-alpha * y_sign * pred_sign)
            w_sum = float(w.sum())
            if w_sum <= 0:
                break
            w = w / w_sum

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            return np.full(X.shape[0], 0.5, dtype=float)

        F = np.zeros(X.shape[0], dtype=float)
        for alpha, est in zip(self.alphas_, self.estimators_):
            y_pred01 = est.predict(X).astype(int)
            pred_sign = 2.0 * y_pred01.astype(float) - 1.0
            if np.isinf(alpha):
                # Perfect separator; return deterministic outcome.
                return (pred_sign > 0).astype(float)
            F += alpha * pred_sign

        # Convert margin to probability via sigmoid(2F)
        return self._sigmoid(2.0 * F)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

