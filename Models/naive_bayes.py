from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .base import BaseBinaryClassifier


class NaiveBayesClassifier(BaseBinaryClassifier):
    """
    Naive Bayes for binary classification.

    Supported distributions:
    - 'gaussian'
    - 'bernoulli' (treat features as binary using `X > 0`)
    - 'multinomial' (treat features as non-negative counts)

    Smoothing:
    - `alpha` acts as an additive prior for bernoulli/multinomial.
    - For gaussian, we add `var_smoothing` to variance.
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        distribution: Literal["gaussian", "bernoulli", "multinomial"] = "gaussian",
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
    ):
        super().__init__(balance=balance)
        self.distribution = str(distribution).lower()
        if self.distribution not in {"gaussian", "bernoulli", "multinomial"}:
            raise ValueError("distribution must be one of {'gaussian','bernoulli','multinomial'}")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        self.var_smoothing = float(var_smoothing)

        # Learned params (class index: 0/1)
        self.class_prior_: Optional[np.ndarray] = None  # shape (2,)

        # gaussian
        self.theta_mean_: Optional[np.ndarray] = None  # (2, d)
        self.theta_var_: Optional[np.ndarray] = None  # (2, d)

        # bernoulli
        self.bernoulli_p_: Optional[np.ndarray] = None  # (2, d) probability of feature=1

        # multinomial
        self.multinomial_log_p_: Optional[np.ndarray] = None  # (2, d) log P(feature)

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        n, d = X.shape
        sw = sample_weight.astype(float, copy=False)
        total_w = float(sw.sum())
        if total_w <= 0:
            sw = np.ones_like(sw)
            total_w = float(sw.sum())

        w0 = float(sw[y_bin == 0].sum())
        w1 = float(sw[y_bin == 1].sum())
        if w0 + w1 <= 0:
            w0 = 1.0
            w1 = 1.0

        self.class_prior_ = np.array([w0 / (w0 + w1), w1 / (w0 + w1)], dtype=float)

        if self.distribution == "gaussian":
            self.theta_mean_ = np.zeros((2, d), dtype=float)
            self.theta_var_ = np.zeros((2, d), dtype=float)

            for cls in (0, 1):
                mask = y_bin == cls
                swc = sw[mask]
                Xc = X[mask]
                denom = float(swc.sum())
                if denom <= 0:
                    self.theta_mean_[cls, :] = 0.0
                    self.theta_var_[cls, :] = 1.0
                    continue
                mean = (Xc * swc[:, None]).sum(axis=0) / denom
                var = (swc[:, None] * (Xc - mean) ** 2).sum(axis=0) / denom
                var = var + self.var_smoothing
                self.theta_mean_[cls, :] = mean
                self.theta_var_[cls, :] = var

        elif self.distribution == "bernoulli":
            # Convert features to {0,1} presence.
            Xb = (X > 0).astype(float)
            self.bernoulli_p_ = np.zeros((2, d), dtype=float)
            for cls in (0, 1):
                mask = y_bin == cls
                swc = sw[mask]
                denom = float(swc.sum())
                if denom <= 0:
                    self.bernoulli_p_[cls, :] = 0.5
                    continue
                # Weighted count of x=1 per feature.
                cnt1 = (Xb[mask] * swc[:, None]).sum(axis=0)
                # Laplace smoothing: alpha pseudo-counts.
                p = (cnt1 + self.alpha) / (denom + 2.0 * self.alpha)
                p = np.clip(p, 1e-12, 1.0 - 1e-12)
                self.bernoulli_p_[cls, :] = p

        else:  # multinomial
            # Ensure non-negative counts.
            Xn = np.maximum(X, 0.0)
            self.multinomial_log_p_ = np.zeros((2, d), dtype=float)
            for cls in (0, 1):
                mask = y_bin == cls
                swc = sw[mask]
                Xc = Xn[mask]
                # Weighted feature sums.
                feature_sum = (Xc * swc[:, None]).sum(axis=0)  # shape (d,)
                total = float(feature_sum.sum())
                if total <= 0:
                    self.multinomial_log_p_[cls, :] = -np.log(d)
                    continue
                p = (feature_sum + self.alpha) / (total + self.alpha * d)
                p = np.clip(p, 1e-300, 1.0)
                self.multinomial_log_p_[cls, :] = np.log(p)

    def _logsumexp(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        amax = np.max(a, axis=axis, keepdims=True)
        out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
        return out.squeeze(axis=axis)

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Returns log p(x|class) + log prior for both classes.
        Output shape: (n_samples, 2)
        """
        assert self.class_prior_ is not None
        log_prior = np.log(self.class_prior_)
        n, d = X.shape

        if self.distribution == "gaussian":
            assert self.theta_mean_ is not None and self.theta_var_ is not None
            out = np.zeros((n, 2), dtype=float)
            for cls in (0, 1):
                mean = self.theta_mean_[cls]
                var = self.theta_var_[cls]
                # log N(x | mean, var) ignoring constant terms would still preserve argmax,
                # but for probabilities keep full (minus constant) form for stability.
                log_prob = -0.5 * (np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var)
                out[:, cls] = log_prior[cls] + log_prob.sum(axis=1)
            return out

        if self.distribution == "bernoulli":
            assert self.bernoulli_p_ is not None
            Xb = (X > 0).astype(float)
            out = np.zeros((n, 2), dtype=float)
            for cls in (0, 1):
                p = self.bernoulli_p_[cls]
                logp = np.log(p)
                log1mp = np.log(1.0 - p)
                out[:, cls] = log_prior[cls] + (Xb * logp + (1.0 - Xb) * log1mp).sum(axis=1)
            return out

        # multinomial
        assert self.multinomial_log_p_ is not None
        Xn = np.maximum(X, 0.0)
        out = np.zeros((n, 2), dtype=float)
        for cls in (0, 1):
            out[:, cls] = log_prior[cls] + (Xn * self.multinomial_log_p_[cls]).sum(axis=1)
        return out

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)  # (n,2)
        # Convert to P(class|x)
        log_denom = self._logsumexp(jll, axis=1)  # (n,)
        log_post_1 = jll[:, 1] - log_denom
        return np.exp(log_post_1)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

