from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .base import BaseBinaryClassifier


class LogisticRegression(BaseBinaryClassifier):
    """
    Regularized binary logistic regression.

    - penalty: 'l1' or 'l2'
    - reg_lambda: regularization strength (applied to weights, not intercept)

    Optimization:
    - L2: (weighted) gradient descent on log-loss + (reg_lambda/2)*||w||^2
    - L1: gradient descent + proximal (soft-thresholding) step on weights
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        penalty: Literal["l1", "l2"] = "l2",
        reg_lambda: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 300,
        lr: float = 0.1,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        super().__init__(balance=balance)
        self.penalty = str(penalty).lower()
        if self.penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be one of {'l1','l2'}")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")
        self.reg_lambda = float(reg_lambda)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.tol = float(tol)
        self.random_state = random_state

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        n, d = X.shape
        sw = sample_weight.astype(float, copy=False)
        total_w = float(sw.sum())
        if total_w <= 0:
            # Degenerate: fall back to uniform.
            sw = np.ones_like(sw)
            total_w = float(sw.sum())

        w = np.zeros(d, dtype=float)
        b = 0.0

        for _ in range(self.max_iter):
            z = X @ w + (b if self.fit_intercept else 0.0)
            p = self._sigmoid(z)
            # Weighted gradient of log-loss.
            # d/dz: (p - y); then chain with w and b.
            err = (p - y_bin.astype(float))
            grad_w = (X.T @ (sw * err)) / total_w
            grad_b = float((sw * err).sum() / total_w)

            if self.penalty == "l2":
                grad_w_reg = grad_w + self.reg_lambda * w
                w_new = w - self.lr * grad_w_reg
            else:
                # Proximal gradient for L1
                w_tmp = w - self.lr * grad_w
                shrink = self.lr * self.reg_lambda
                w_new = np.sign(w_tmp) * np.maximum(0.0, np.abs(w_tmp) - shrink)

            if self.fit_intercept:
                b_new = b - self.lr * grad_b
            else:
                b_new = 0.0

            # Convergence: relative parameter movement.
            delta = np.linalg.norm(w_new - w)
            w = w_new
            b = b_new
            if delta <= self.tol:
                break

        self.w_ = w
        self.b_ = float(b)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Internal error: w_ missing; did fit() run?")
        z = X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
        return self._sigmoid(z)

