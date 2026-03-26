from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _as_2d_float_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (n_samples, n_features); got {X.shape}")
    return X.astype(float, copy=False)


def _as_1d_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    return y


class BaseBinaryClassifier(abc.ABC):
    """
    Shared utilities for binary classification models.

    Subclasses must implement:
    - `_fit_internal(X, y_bin, sample_weight)`
    - `_predict_internal(X) -> y_bin`
    - `_predict_proba_internal(X) -> proba for class 1`
    """

    def __init__(self, *, balance: bool = False):
        self.balance = bool(balance)
        self.classes_: Optional[np.ndarray] = None  # shape (2,)
        self._y_bin_train: Optional[np.ndarray] = None

    # --------------------
    # Public sklearn-like API
    # --------------------
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = _as_2d_float_array(X)
        y = _as_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape} vs {y.shape}")

        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"Binary classification expected exactly 2 classes; got {classes.shape[0]}: {classes}"
            )

        self.classes_ = classes
        y_bin = (y == classes[1]).astype(int)
        self._y_bin_train = y_bin

        sw = self._compute_sample_weight(y_bin, sample_weight=sample_weight)
        self._fit_internal(X, y_bin, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = _as_2d_float_array(X)
        y_bin = self._predict_internal(X)
        return self._decode_y_bin(y_bin)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = _as_2d_float_array(X)
        proba1 = self._predict_proba_internal(X)
        proba1 = np.clip(proba1, 0.0, 1.0)
        proba0 = 1.0 - proba1
        return np.column_stack([proba0, proba1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = _as_1d_array(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    # --------------------
    # Utilities for subclasses
    # --------------------
    def _check_is_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _decode_y_bin(self, y_bin: np.ndarray) -> np.ndarray:
        classes = self.classes_
        assert classes is not None
        y_bin = np.asarray(y_bin).reshape(-1)
        if not np.issubdtype(y_bin.dtype, np.integer):
            y_bin = (y_bin >= 0.5).astype(int)
        return np.where(y_bin == 1, classes[1], classes[0])

    def _encode_y_bin(self, y: np.ndarray) -> np.ndarray:
        classes = self.classes_
        assert classes is not None
        return (y == classes[1]).astype(int)

    def _compute_sample_weight(self, y_bin: np.ndarray, sample_weight: Optional[np.ndarray]) -> np.ndarray:
        """
        If `balance=True`, reweight classes so each class has approximately equal total weight.
        """
        n = y_bin.shape[0]
        if sample_weight is None:
            sw = np.ones(n, dtype=float)
        else:
            sw = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sw.shape[0] != n:
                raise ValueError(f"sample_weight has length {sw.shape[0]} but expected {n}")

        if not self.balance:
            return sw

        total = float(sw.sum())
        if total <= 0:
            return sw

        w0 = float(sw[y_bin == 0].sum())
        w1 = float(sw[y_bin == 1].sum())
        # Avoid division by zero for degenerate training sets.
        denominator0 = w0 if w0 > 0 else 1.0
        denominator1 = w1 if w1 > 0 else 1.0
        # Target equal total weight per class.
        # class 0 multiplier: total/(2*w0), class 1 multiplier: total/(2*w1)
        m0 = total / (2.0 * denominator0)
        m1 = total / (2.0 * denominator1)
        sw_balanced = np.where(y_bin == 0, sw * m0, sw * m1)
        return sw_balanced

    # --------------------
    # Shared math helpers
    # --------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically-stable sigmoid
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    # --------------------
    # Abstract methods
    # --------------------
    @abc.abstractmethod
    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

