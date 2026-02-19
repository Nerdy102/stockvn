from __future__ import annotations

import numpy as np


def var_historical(returns, alpha: float = 0.05) -> float:
    x = np.asarray(returns, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, alpha))


def es_historical(returns, alpha: float = 0.05) -> float:
    x = np.asarray(returns, dtype=float)
    if x.size == 0:
        return 0.0
    var = var_historical(x, alpha)
    tail = x[x <= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))
