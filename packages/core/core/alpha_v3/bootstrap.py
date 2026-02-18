from __future__ import annotations

import math

import numpy as np
import pandas as pd


def block_bootstrap_ci(
    returns: pd.Series,
    block: int = 20,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) == 0:
        return {"sharpe_lo": 0.0, "sharpe_hi": 0.0, "cagr_lo": 0.0, "cagr_hi": 0.0}

    rng = np.random.default_rng(seed)
    blocks = max(1, int(block))
    n_blk = math.ceil(len(r) / blocks)
    sharpe_vals: list[float] = []
    cagr_vals: list[float] = []

    for _ in range(int(n_resamples)):
        picks: list[int] = []
        for _ in range(n_blk):
            start = int(rng.integers(0, max(1, len(r) - blocks + 1)))
            picks.extend(range(start, min(len(r), start + blocks)))
        sample = r[np.array(picks[: len(r)], dtype=int)]
        sharpe = float((sample.mean() / (sample.std(ddof=0) + 1e-12)) * math.sqrt(252.0))
        years = max(len(sample) / 252.0, 1e-12)
        equity = float(np.prod(1.0 + sample))
        cagr = -1.0 if equity <= 0 else float(equity ** (1.0 / years) - 1.0)
        sharpe_vals.append(sharpe)
        cagr_vals.append(cagr)

    return {
        "sharpe_lo": float(np.quantile(sharpe_vals, 0.025)),
        "sharpe_hi": float(np.quantile(sharpe_vals, 0.975)),
        "cagr_lo": float(np.quantile(cagr_vals, 0.025)),
        "cagr_hi": float(np.quantile(cagr_vals, 0.975)),
    }
