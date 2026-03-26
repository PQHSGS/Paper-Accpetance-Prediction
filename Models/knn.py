from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .base import BaseBinaryClassifier


class KNNClassifier(BaseBinaryClassifier):
    """
    K-Nearest-Neighbors classifier (binary) for dense numpy arrays.

    Parameters:
    - k: number of neighbors
    - metric: 'euclidean', 'manhattan', 'cosine', 'minkowski'
    - p: Minkowski parameter (for metric='minkowski')
    - weights:
        - 'uniform': neighbor vote weight = sample_weight
        - 'distance': neighbor vote weight = sample_weight / (dist + eps)
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        k: int = 5,
        metric: Literal["euclidean", "manhattan", "cosine", "minkowski"] = "euclidean",
        p: float = 2.0,
        weights: Literal["uniform", "distance"] = "uniform",
        eps: float = 1e-9,
    ):
        super().__init__(balance=balance)
        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be >= 1")
        self.metric = str(metric).lower()
        if self.metric not in {"euclidean", "manhattan", "cosine", "minkowski"}:
            raise ValueError("metric must be one of {'euclidean','manhattan','cosine','minkowski'}")
        self.p = float(p)
        self.weights = str(weights).lower()
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be one of {'uniform','distance'}")
        self.eps = float(eps)

        self._X_train: Optional[np.ndarray] = None
        self._y_bin_train: Optional[np.ndarray] = None
        self._sw_train: Optional[np.ndarray] = None

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        self._X_train = X.astype(float, copy=False)
        self._y_bin_train = y_bin.astype(int, copy=False)
        self._sw_train = sample_weight.astype(float, copy=False)

    def _distance_to_train(self, x: np.ndarray) -> np.ndarray:
        assert self._X_train is not None
        assert self._y_bin_train is not None

        Xtr = self._X_train
        if self.metric == "euclidean":
            d2 = np.sum(Xtr * Xtr, axis=1) + np.sum(x * x) - 2.0 * (Xtr @ x)
            return np.sqrt(np.maximum(d2, 0.0))
        if self.metric == "manhattan":
            return np.sum(np.abs(Xtr - x[None, :]), axis=1)
        if self.metric == "minkowski":
            return np.sum(np.abs(Xtr - x[None, :]) ** self.p, axis=1) ** (1.0 / self.p)
        # cosine distance
        x_norm = float(np.linalg.norm(x))
        tr_norm = np.linalg.norm(Xtr, axis=1)
        denom = tr_norm * x_norm
        # If either is zero, define cosine similarity as 0 -> cosine distance=1.
        sim = np.zeros_like(denom, dtype=float)
        nonzero = denom > 0
        sim[nonzero] = (Xtr[nonzero] @ x) / denom[nonzero]
        return 1.0 - sim

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        assert self._X_train is not None
        assert self._y_bin_train is not None
        assert self._sw_train is not None

        n_test = X.shape[0]
        proba1 = np.zeros(n_test, dtype=float)

        k = min(self.k, self._X_train.shape[0])
        for i in range(n_test):
            x = X[i]
            dist = self._distance_to_train(x)
            # Fast partial selection
            idx = np.argpartition(dist, k - 1)[:k]
            d_sel = dist[idx]
            y_sel = self._y_bin_train[idx]
            w_sel = self._sw_train[idx]

            if self.weights == "distance":
                w_sel = w_sel / (d_sel + self.eps)

            denom = float(w_sel.sum())
            if denom <= 0:
                proba1[i] = 0.5
            else:
                proba1[i] = float(w_sel[y_sel == 1].sum()) / denom
        return proba1

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

