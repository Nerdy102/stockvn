from __future__ import annotations

import numpy as np

from .hrp import hrp_weights


def equal_weight(assets: list[str]) -> dict[str, float]:
    n = max(len(assets), 1)
    w = 1.0 / n
    return {a: w for a in assets}


def inverse_vol(returns_matrix, assets: list[str]) -> dict[str, float]:
    x = np.asarray(returns_matrix, dtype=float)
    vol = np.std(x, axis=0, ddof=1)
    inv = 1.0 / np.clip(vol, 1e-12, None)
    w = inv / np.sum(inv)
    return {a: float(w[i]) for i, a in enumerate(assets)}


def hrp(returns_matrix, assets: list[str]) -> dict[str, float]:
    w = hrp_weights(returns_matrix)
    return {a: float(w[i]) for i, a in enumerate(assets)}
