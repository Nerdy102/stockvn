from __future__ import annotations

import numpy as np
import pandas as pd


def _stationary_bootstrap_indices(t: int, p: float, rng: np.random.Generator) -> np.ndarray:
    if t <= 0:
        return np.array([], dtype=int)
    idx = np.empty(t, dtype=int)
    idx[0] = int(rng.integers(0, t))
    for i in range(1, t):
        if float(rng.random()) < p:
            idx[i] = int(rng.integers(0, t))
        else:
            idx[i] = (idx[i - 1] + 1) % t
    return idx


def white_reality_check(
    benchmark_returns: pd.Series,
    competitor_returns: pd.DataFrame,
    n_bootstrap: int = 1000,
    block_mean: float = 20.0,
    seed: int = 42,
) -> tuple[float, dict[str, float]]:
    """
    White Reality Check on maximum competitor-minus-benchmark mean return delta.

    Null: best competitor is not better than benchmark.
    """
    bench = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    comp = pd.DataFrame(competitor_returns).astype(float).replace([np.inf, -np.inf], np.nan)

    aligned = comp.copy()
    aligned["__bench__"] = bench
    aligned = aligned.dropna(axis=0, how="any")
    if aligned.empty or comp.shape[1] == 0:
        return 1.0, {"t": 0.0, "n_competitors": float(comp.shape[1]), "obs_stat": 0.0}

    bench_a = aligned.pop("__bench__")
    deltas = aligned.sub(bench_a, axis=0)

    t = len(deltas)
    n_comp = deltas.shape[1]
    obs_means = deltas.mean(axis=0)
    obs_stat = float(np.max(obs_means.values))

    centered = deltas - obs_means
    p = min(max(1.0 / max(block_mean, 1.0), 1e-6), 1.0)
    rng = np.random.default_rng(seed)

    boot_stats = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(t, p=p, rng=rng)
        boot_sample = centered.iloc[idx]
        boot_stats[b] = float(np.max(boot_sample.mean(axis=0).values))

    p_val = float((np.sum(boot_stats >= obs_stat) + 1.0) / (n_bootstrap + 1.0))
    return p_val, {
        "t": float(t),
        "n_competitors": float(n_comp),
        "obs_stat": float(obs_stat),
        "block_mean": float(block_mean),
        "n_bootstrap": float(n_bootstrap),
    }
