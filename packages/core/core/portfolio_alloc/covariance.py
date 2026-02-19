from __future__ import annotations

import numpy as np


def sample_cov(returns_matrix) -> np.ndarray:
    x = np.asarray(returns_matrix, dtype=float)
    return np.cov(x, rowvar=False, ddof=1)


def ewma_cov(returns_matrix, lambda_: float = 0.94) -> np.ndarray:
    x = np.asarray(returns_matrix, dtype=float)
    t, n = x.shape
    s = np.cov(x[: min(20, t)], rowvar=False, ddof=1)
    for i in range(min(20, t), t):
        v = x[i].reshape(-1, 1)
        s = lambda_ * s + (1.0 - lambda_) * (v @ v.T)
    return s
