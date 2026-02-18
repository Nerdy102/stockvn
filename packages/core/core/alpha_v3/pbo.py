from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def _annualized_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    return float((r.mean() / (r.std(ddof=0) + 1e-12)) * math.sqrt(252.0))


def _rank_percentile(values: pd.Series, value: float) -> float:
    # percentile rank in (0,1)
    pct = float((values <= value).sum()) / float(max(len(values), 1))
    return float(np.clip(pct, 1e-6, 1 - 1e-6))


def compute_pbo_cscv(variant_returns: pd.DataFrame, slices: int = 10) -> tuple[float, dict[str, float]]:
    """Compute PBO via CSCV with contiguous slices and combinatorial IS/OOS splits."""
    if variant_returns.empty or variant_returns.shape[1] < 2:
        return 1.0, {"n_logits": 0.0, "omega_mean": 0.0, "logit_mean": 0.0}

    data = variant_returns.copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx_slices = np.array_split(np.arange(len(data)), slices)
    if any(len(x) == 0 for x in idx_slices):
        return 1.0, {"n_logits": 0.0, "omega_mean": 0.0, "logit_mean": 0.0}

    logits: list[float] = []
    omegas: list[float] = []
    half = slices // 2
    for is_idx in itertools.combinations(range(slices), half):
        oos_idx = [i for i in range(slices) if i not in is_idx]
        is_rows = np.concatenate([idx_slices[i] for i in is_idx])
        oos_rows = np.concatenate([idx_slices[i] for i in oos_idx])

        is_sharpes = data.iloc[is_rows].apply(_annualized_sharpe, axis=0)
        best_variant = str(is_sharpes.idxmax())

        oos_sharpes = data.iloc[oos_rows].apply(_annualized_sharpe, axis=0)
        omega = _rank_percentile(oos_sharpes, float(oos_sharpes.loc[best_variant]))
        logit = float(math.log(omega / (1.0 - omega)))
        omegas.append(float(omega))
        logits.append(logit)

    logits_arr = np.array(logits, dtype=float)
    phi = float(np.mean(logits_arr < 0.0)) if len(logits_arr) else 1.0
    summary = {
        "n_logits": float(len(logits)),
        "omega_mean": float(np.mean(omegas)) if omegas else 0.0,
        "omega_median": float(np.median(omegas)) if omegas else 0.0,
        "logit_mean": float(np.mean(logits_arr)) if len(logits_arr) else 0.0,
        "logit_median": float(np.median(logits_arr)) if len(logits_arr) else 0.0,
        "logit_min": float(np.min(logits_arr)) if len(logits_arr) else 0.0,
        "logit_max": float(np.max(logits_arr)) if len(logits_arr) else 0.0,
        "slices": float(slices),
    }
    return phi, summary
