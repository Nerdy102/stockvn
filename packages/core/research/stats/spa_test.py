from __future__ import annotations

import numpy as np
import pandas as pd

from research.stats.reality_check import _stationary_bootstrap_indices


def hansen_spa_test(
    benchmark_returns: pd.Series,
    competitor_returns: pd.DataFrame,
    n_bootstrap: int = 1000,
    block_mean: float = 20.0,
    seed: int = 42,
) -> tuple[float, dict[str, float]]:
    """
    Hansen SPA test with studentized max statistic and sample-dependent null.

    Null: no competitor has positive expected performance over benchmark.
    """
    bench = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    comp = pd.DataFrame(competitor_returns).astype(float).replace([np.inf, -np.inf], np.nan)

    aligned = comp.copy()
    aligned["__bench__"] = bench
    aligned = aligned.dropna(axis=0, how="any")
    if aligned.empty or comp.shape[1] == 0:
        return 1.0, {"t": 0.0, "n_competitors": float(comp.shape[1]), "obs_stat": 0.0}

    bench_a = aligned.pop("__bench__")
    d = aligned.sub(bench_a, axis=0)
    t = len(d)
    n_comp = d.shape[1]

    mu_hat = d.mean(axis=0).values
    sigma = d.std(axis=0, ddof=1).values
    sigma = np.maximum(sigma, 1e-12)

    # Studentized observed statistic
    z = np.sqrt(t) * mu_hat / sigma
    obs_stat = float(np.max(z))

    # Sample-dependent null: only keep positive estimated means in recentering term.
    mu_pos = np.maximum(mu_hat, 0.0)

    p = min(max(1.0 / max(block_mean, 1.0), 1e-6), 1.0)
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=float)

    vals = d.values
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(t, p=p, rng=rng)
        bs = vals[idx, :]

        # Sample-dependent recentering under null
        mu_bs = bs.mean(axis=0) - mu_pos
        z_bs = np.sqrt(t) * mu_bs / sigma
        boot_stats[b] = float(np.max(z_bs))

    p_val = float((np.sum(boot_stats >= obs_stat) + 1.0) / (n_bootstrap + 1.0))
    return p_val, {
        "t": float(t),
        "n_competitors": float(n_comp),
        "obs_stat": float(obs_stat),
        "block_mean": float(block_mean),
        "n_bootstrap": float(n_bootstrap),
    }
