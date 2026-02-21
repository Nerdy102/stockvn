from __future__ import annotations

import numpy as np
import pandas as pd

from research.stats.reality_check import _circular_block_indices, white_reality_check


def hansen_spa_test(
    benchmark: pd.Series,
    competitors: pd.DataFrame,
    n_bootstrap: int = 2000,
    block_mean: float = 20.0,
    seed: int = 42,
) -> tuple[float, dict[str, float]]:
    rc_p, rc_comp = white_reality_check(
        benchmark=benchmark,
        competitors=competitors,
        n_bootstrap=n_bootstrap,
        block_mean=block_mean,
        seed=seed,
    )

    bench = benchmark.astype(float).reset_index(drop=True)
    comp = competitors.astype(float).reset_index(drop=True)
    diff = (
        comp.sub(bench, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    )
    if diff.ndim == 1:
        diff = diff[:, None]
    n = int(diff.shape[0])
    if n == 0:
        return rc_p, rc_comp

    mu = diff.mean(axis=0)
    centered = diff - np.minimum(mu, 0.0)
    obs = float(np.sqrt(n) * np.max(np.maximum(mu, 0.0)))

    block = max(1, int(round(block_mean)))
    rng = np.random.default_rng(seed + 17)
    stats = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = _circular_block_indices(n, block, rng)
        sample = centered[idx, :] - centered.mean(axis=0, keepdims=True)
        stats[b] = float(np.sqrt(n) * np.max(np.maximum(sample.mean(axis=0), 0.0)))

    spa_p = float((1.0 + np.sum(stats >= obs)) / (n_bootstrap + 1.0))
    return min(spa_p, rc_p), {"obs": obs, "bootstrap_B": float(n_bootstrap)}
