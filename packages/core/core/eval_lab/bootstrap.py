from __future__ import annotations

import math

import numpy as np


def block_bootstrap_samples(
    values: list[float],
    n_samples: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    rng = np.random.default_rng(seed)
    blocks = max(1, math.ceil(n / block_size))
    out = np.zeros((n_samples, n), dtype=float)
    for b in range(n_samples):
        picks = []
        for _ in range(blocks):
            i = int(rng.integers(0, max(1, n - block_size + 1)))
            picks.append(arr[i : i + block_size])
        out[b, :] = np.concatenate(picks)[:n]
    return out


def bootstrap_ci(
    values: list[float], seed: int, n_samples: int, block_size: int
) -> dict[str, float]:
    sam = block_bootstrap_samples(values, n_samples=n_samples, block_size=block_size, seed=seed)
    if sam.size == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    metric = np.mean(sam, axis=1)
    return {
        "p05": float(np.quantile(metric, 0.05)),
        "p50": float(np.quantile(metric, 0.50)),
        "p95": float(np.quantile(metric, 0.95)),
    }
