from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .base import BaseBinaryClassifier
from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier(BaseBinaryClassifier):
    """
    Random Forest for binary classification (dense numpy arrays).

    Implemented as:
    - bagging of weighted decision trees
    - each tree sees a bootstrap sample of rows (optionally weighted by sample_weight)
    - each split considers `max_features` randomly (delegated to the DecisionTreeClassifier)
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        n_estimators: int = 100,
        bootstrap: bool = True,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        random_state: Optional[int] = None,
        # Base tree params
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        super().__init__(balance=balance)
        self.n_estimators = int(n_estimators)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.bootstrap = bool(bootstrap)
        self.max_features = max_features
        self.random_state = random_state

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.trees_: Optional[list[DecisionTreeClassifier]] = None

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        n = X.shape[0]
        sw = sample_weight.astype(float, copy=False)
        total = float(sw.sum())
        if total <= 0:
            sw = np.ones_like(sw, dtype=float)
            total = float(sw.sum())
        probs = sw / total

        rng = np.random.default_rng(self.random_state)
        trees: list[DecisionTreeClassifier] = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.choice(n, size=n, replace=True, p=probs)
            else:
                idx = np.arange(n, dtype=int)

            Xb = X[idx]
            yb = y_bin[idx]
            swb = sw[idx]

            tree = DecisionTreeClassifier(
                balance=False,  # balance already applied in BaseBinaryClassifier
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=None if self.random_state is None else (self.random_state + i),
            )
            tree.fit(Xb, yb, sample_weight=swb)
            trees.append(tree)

        self.trees_ = trees

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        assert self.trees_ is not None
        n = X.shape[0]
        proba1 = np.zeros(n, dtype=float)
        for tree in self.trees_:
            p = tree.predict_proba(X)[:, 1]
            proba1 += p
        proba1 /= len(self.trees_)
        return proba1

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

