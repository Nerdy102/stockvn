from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .sharpe import sharpe_non_annualized


def _metric(metric_name: str, arr: np.ndarray) -> float:
    if metric_name == "net_return":
        return float(np.prod(1.0 + arr) - 1.0)
    if metric_name == "sharpe_non_annualized":
        return float(sharpe_non_annualized(arr))
    raise ValueError("metric không hỗ trợ")


def block_bootstrap_ci(
    returns,
    metric_fn: str,
    block_size: int,
    n_iter: int,
    ci: tuple[float, float] = (0.05, 0.95),
    seed: int = 42,
) -> dict:
    n_iter = min(max(int(n_iter), 1), 500)
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    t = int(arr.size)
    if t < 3 * int(block_size):
        return {"ci": None, "reason": "sample quá ngắn", "n_iter": n_iter}

    rng = np.random.default_rng(seed)
    vals = []
    blocks = max(1, math.ceil(t / block_size))
    for _ in range(n_iter):
        pieces = []
        for _ in range(blocks):
            i = int(rng.integers(0, max(1, t - block_size + 1)))
            pieces.append(arr[i : i + block_size])
        sample = np.concatenate(pieces)[:t]
        vals.append(_metric(metric_fn, sample))
    vals = np.sort(np.asarray(vals, dtype=float))
    p05 = float(np.quantile(vals, ci[0]))
    p50 = float(np.quantile(vals, 0.5))
    p95 = float(np.quantile(vals, ci[1]))
    return {"ci": {"p05": p05, "p50": p50, "p95": p95}, "reason": "OK", "n_iter": n_iter}
