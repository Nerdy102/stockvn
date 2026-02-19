from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return a[np.isfinite(a)]


def sample_mean(x) -> float:
    a = _arr(x)
    if a.size == 0:
        return 0.0
    return float(np.mean(a))


def sample_std(x, ddof: int = 1) -> float:
    a = _arr(x)
    if a.size <= ddof:
        return 0.0
    return float(np.std(a, ddof=ddof))


def sample_skewness_gamma3(x) -> float:
    a = _arr(x)
    if a.size < 3:
        return 0.0
    mu = float(np.mean(a))
    sigma = float(np.std(a, ddof=1))
    if sigma <= 0:
        return 0.0
    return float(np.mean(((a - mu) / sigma) ** 3))


def sample_kurtosis_gamma4(x) -> float:
    a = _arr(x)
    if a.size < 4:
        return 3.0
    mu = float(np.mean(a))
    sigma = float(np.std(a, ddof=1))
    if sigma <= 0:
        return 3.0
    return float(np.mean(((a - mu) / sigma) ** 4))
