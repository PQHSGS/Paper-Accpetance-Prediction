from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

from .base import BaseBinaryClassifier


class SVMClassifier(BaseBinaryClassifier):
    """
    From-scratch binary SVM.

    - kernel='linear': Pegasos-style primal solver (fast(er), no guarantee of exact optimum).
    - kernel in {'rbf','poly','sigmoid'}: simplified SMO-style solver (slower; intended for smaller datasets).

    Probability estimation:
    - Uses a monotonic mapping `P(y=1|x) = sigmoid(probability_scale * decision_function(x))`.
      This is not calibrated like Platt scaling.
    """

    def __init__(
        self,
        *,
        balance: bool = False,
        kernel: Literal["linear", "rbf", "poly", "sigmoid"] = "linear",
        C: float = 1.0,
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 0.0,
        # Linear (Pegasos) params
        max_iter_linear: int = 2000,
        # Kernel (SMO) params
        max_passes: int = 5,
        tol: float = 1e-3,
        eps_alpha: float = 1e-5,
        # Bias handling: for Pegasos, we learn bias by augmenting X with a constant 1 feature.
        fit_intercept: bool = True,
        # Approximate handling of per-sample weights by weighted resampling.
        use_sample_weight_resample: bool = True,
        resample_n: Optional[int] = None,
        probability_scale: float = 1.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(balance=balance)
        self.kernel = str(kernel).lower()
        if self.kernel not in {"linear", "rbf", "poly", "sigmoid"}:
            raise ValueError("kernel must be one of {'linear','rbf','poly','sigmoid'}")
        if C <= 0:
            raise ValueError("C must be > 0")
        self.C = float(C)
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.max_iter_linear = int(max_iter_linear)
        self.max_passes = int(max_passes)
        self.tol = float(tol)
        self.eps_alpha = float(eps_alpha)
        self.fit_intercept = bool(fit_intercept)
        self.use_sample_weight_resample = bool(use_sample_weight_resample)
        self.resample_n = resample_n
        self.probability_scale = float(probability_scale)
        self.random_state = random_state

        # Learned params:
        # Linear
        self.w_: Optional[np.ndarray] = None
        # Kernel
        self.alphas_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self._sv_X: Optional[np.ndarray] = None
        self._sv_y: Optional[np.ndarray] = None  # {-1,+1}

        # Augmentation info for Pegasos
        self._trained_with_bias_: bool = False

    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # x1 shape (n_sv, d), x2 shape (d,) or (n, d) -> return (n_sv, n)
        assert x1.ndim == 2
        if self.gamma is None:
            gamma = 1.0 / x1.shape[1]
        else:
            gamma = float(self.gamma)

        if self.kernel == "linear":
            return x1 @ x2.T if x2.ndim == 2 else x1 @ x2

        if x2.ndim == 1:
            x2_2d = x2[None, :]
        else:
            x2_2d = x2

        if self.kernel == "rbf":
            # ||x - x'||^2 = ||x||^2 + ||x'||^2 -2 x·x'
            x1_norm = np.sum(x1 * x1, axis=1)[:, None]
            x2_norm = np.sum(x2_2d * x2_2d, axis=1)[None, :]
            sqdist = x1_norm + x2_norm - 2.0 * (x1 @ x2_2d.T)
            sqdist = np.maximum(sqdist, 0.0)
            return np.exp(-gamma * sqdist)

        if self.kernel == "poly":
            return (gamma * (x1 @ x2_2d.T) + self.coef0) ** self.degree

        # sigmoid
        return np.tanh(gamma * (x1 @ x2_2d.T) + self.coef0)

    def _maybe_resample(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_sample_weight_resample:
            return X, y_bin
        n = X.shape[0]
        sw = sample_weight.astype(float, copy=False)
        total = float(sw.sum())
        if total <= 0:
            probs = np.full(n, 1.0 / n, dtype=float)
        else:
            probs = sw / total
        m = self.resample_n if self.resample_n is not None else n
        m = int(m)
        if m <= 0:
            return X, y_bin
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=m, replace=True, p=probs)
        return X[idx], y_bin[idx]

    def _fit_linear_pegasos(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        # Resample to incorporate sample weights approximately.
        Xr, y_bin_r = self._maybe_resample(X, y_bin, sample_weight)
        y_sign = 2.0 * y_bin_r.astype(float) - 1.0

        if np.all(y_sign == -1.0) or np.all(y_sign == 1.0):
            # Degenerate: constant classifier.
            self.w_ = np.zeros(Xr.shape[1] + (1 if self.fit_intercept else 0), dtype=float)
            self.b_ = -1.0 if np.all(y_sign == -1.0) else 1.0
            self._trained_with_bias_ = self.fit_intercept
            return

        if self.fit_intercept:
            X_aug = np.concatenate([Xr, np.ones((Xr.shape[0], 1), dtype=float)], axis=1)
            self._trained_with_bias_ = True
        else:
            X_aug = Xr
            self._trained_with_bias_ = False

        n, d = X_aug.shape

        # Regularization strength for Pegasos-style update.
        # We map C to lambda via lambda = 1/C (common heuristic).
        lam = 1.0 / max(self.C, 1e-12)
        lam = float(lam)

        rng = np.random.default_rng(self.random_state)
        w = np.zeros(d, dtype=float)

        # Projection radius for soft-margin SVM in Pegasos.
        limit = 1.0 / np.sqrt(lam) if lam > 0 else float("inf")

        for t in range(1, self.max_iter_linear + 1):
            i = int(rng.integers(0, n))
            xi = X_aug[i]
            yi = y_sign[i]
            margin = yi * float(xi @ w)
            eta = 1.0 / (lam * t) if lam > 0 else 0.0

            if margin < 1.0:
                w = (1.0 - 1.0 / t) * w + eta * yi * xi
            else:
                w = (1.0 - 1.0 / t) * w

            if limit != float("inf"):
                norm_w = float(np.linalg.norm(w))
                if norm_w > limit and norm_w > 0:
                    w = w * (limit / norm_w)

        self.w_ = w
        self.alphas_ = None
        self._sv_X = None
        self._sv_y = None
        self.b_ = 0.0

    def _fit_kernel_smo(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        Xr, y_bin_r = self._maybe_resample(X, y_bin, sample_weight)
        n, d = Xr.shape
        y = 2.0 * y_bin_r.astype(float) - 1.0  # {-1,+1}

        if np.all(y == -1.0) or np.all(y == 1.0):
            # Degenerate constant classifier.
            self.w_ = None
            self.alphas_ = None
            self._sv_X = None
            self._sv_y = None
            self.b_ = -1.0 if np.all(y == -1.0) else 1.0
            return

        if self.gamma is None:
            gamma = 1.0 / d
        else:
            gamma = float(self.gamma)

        # Precompute kernel matrix if feasible.
        precompute = n <= 2000
        if precompute:
            # Build K_ij
            if self.kernel == "linear":
                K = Xr @ Xr.T
            elif self.kernel == "rbf":
                x_norm = np.sum(Xr * Xr, axis=1)
                sqdist = x_norm[:, None] + x_norm[None, :] - 2.0 * (Xr @ Xr.T)
                sqdist = np.maximum(sqdist, 0.0)
                K = np.exp(-gamma * sqdist)
            elif self.kernel == "poly":
                K = (gamma * (Xr @ Xr.T) + self.coef0) ** self.degree
            else:  # sigmoid
                K = np.tanh(gamma * (Xr @ Xr.T) + self.coef0)
        else:
            K = None

        def get_K(i: int, j: int) -> float:
            if precompute:
                return float(K[i, j])  # type: ignore[index]
            # Compute on the fly
            x1 = Xr[i : i + 1]
            x2 = Xr[j]
            return float(self._kernel_function(x1, x2))

        alphas = np.zeros(n, dtype=float)
        b = 0.0
        # Cache decision values on training points:
        f = np.zeros(n, dtype=float)  # Σ alpha*y*K + b

        passes = 0
        rng = np.random.default_rng(self.random_state)

        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                Ei = f[i] - y[i]
                # KKT violation checks
                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (y[i] * Ei > self.tol and alphas[i] > 0):
                    # Pick j != i
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                    Ej = f[j] - y[j]

                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    if y[i] != y[j]:
                        L = max(0.0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    if abs(L - H) < 1e-12:
                        continue

                    Ki_i = get_K(i, i)
                    Kj_j = get_K(j, j)
                    Ki_j = get_K(i, j)
                    eta = 2.0 * Ki_j - Ki_i - Kj_j
                    if eta >= 0:
                        continue

                    alpha_j_new = alpha_j_old - (y[j] * (Ei - Ej)) / eta
                    # Clip
                    alpha_j_new = min(H, max(L, alpha_j_new))

                    if abs(alpha_j_new - alpha_j_old) < self.eps_alpha:
                        continue

                    alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

                    # Compute b candidates
                    b_old = b
                    b1 = (
                        b_old
                        - Ei
                        - y[i] * (alpha_i_new - alpha_i_old) * Ki_i
                        - y[j] * (alpha_j_new - alpha_j_old) * Ki_j
                    )
                    b2 = (
                        b_old
                        - Ej
                        - y[i] * (alpha_i_new - alpha_i_old) * Ki_j
                        - y[j] * (alpha_j_new - alpha_j_old) * Kj_j
                    )
                    if 0 < alpha_i_new < self.C:
                        b = b1
                    elif 0 < alpha_j_new < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    delta_b = b - b_old

                    # Update cached f vector:
                    # f <- f + y_i*(a_i_new-a_i_old)*K[:,i] + y_j*(a_j_new-a_j_old)*K[:,j] + delta_b
                    da_i = alpha_i_new - alpha_i_old
                    da_j = alpha_j_new - alpha_j_old

                    if precompute:
                        # K[:, i], K[:, j] from precomputed matrix
                        f = f + y[i] * da_i * K[:, i] + y[j] * da_j * K[:, j] + delta_b
                    else:
                        # Update on the fly (O(n) kernel evaluations per update).
                        for k in range(n):
                            Kki = get_K(k, i)
                            Kkj = get_K(k, j)
                            f[k] = f[k] + y[i] * da_i * Kki + y[j] * da_j * Kkj + delta_b

                    alphas[i] = alpha_i_new
                    alphas[j] = alpha_j_new
                    num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # Store support vectors
        sv = alphas > self.eps_alpha
        self.alphas_ = alphas[sv]
        self.b_ = float(b)
        self._sv_X = Xr[sv]
        self._sv_y = y[sv]
        self.w_ = None

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        if self.kernel == "linear":
            self._fit_linear_pegasos(X, y_bin, sample_weight)
        else:
            self._fit_kernel_smo(X, y_bin, sample_weight)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            assert self.w_ is not None
            if self.fit_intercept:
                X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=float)], axis=1)
            else:
                X_aug = X
            return X_aug @ self.w_ + float(self.b_)

        # kernel decision function uses support vectors
        if self._sv_X is None or self._sv_y is None or self.alphas_ is None:
            # Degenerate fallback: constant margin stored in b_
            return np.full(X.shape[0], float(self.b_), dtype=float)

        # Compute K(sv, X)
        K_sv_X = self._kernel_function(self._sv_X, X)  # (n_sv, n_test)
        # decision = Σ alpha_i*y_i*K_i(x) + b
        return (self.alphas_ * self._sv_y) @ K_sv_X + float(self.b_)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        f = self._decision_function(X)
        # Map margin to probability.
        return self._sigmoid(self.probability_scale * f)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        proba1 = self._predict_proba_internal(X)
        return (proba1 >= 0.5).astype(int)

