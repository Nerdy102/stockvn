from __future__ import annotations

import numpy as np
import pandas as pd


def _circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    idx: list[int] = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend(((start + k) % n) for k in range(block))
    return np.asarray(idx[:n], dtype=int)


def white_reality_check(
    benchmark: pd.Series,
    competitors: pd.DataFrame,
    n_bootstrap: int = 2000,
    block_mean: float = 20.0,
    seed: int = 42,
) -> tuple[float, dict[str, float]]:
    bench = benchmark.astype(float).reset_index(drop=True)
    comp = competitors.astype(float).reset_index(drop=True)
    frame = comp.sub(bench, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diff = frame.to_numpy(dtype=float)
    if diff.ndim == 1:
        diff = diff[:, None]
    n = int(diff.shape[0])
    if n == 0:
        return 1.0, {"obs": 0.0}

    obs = float(np.sqrt(n) * np.max(diff.mean(axis=0)))
    centered = diff - diff.mean(axis=0, keepdims=True)
    block = max(1, int(round(block_mean)))
    rng = np.random.default_rng(seed)

    stats = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = _circular_block_indices(n, block, rng)
        sample = centered[idx, :]
        stats[b] = float(np.sqrt(n) * np.max(sample.mean(axis=0)))

    p = float((1.0 + np.sum(stats >= obs)) / (n_bootstrap + 1.0))
    return p, {"obs": obs, "bootstrap_B": float(n_bootstrap)}
