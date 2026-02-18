from __future__ import annotations

import numpy as np
import pandas as pd


def ndcg_at_k(gain: np.ndarray, score: np.ndarray, k: int = 30) -> float:
    g = np.asarray(gain, dtype=float)
    s = np.asarray(score, dtype=float)
    if g.size == 0:
        return 0.0
    kk = int(max(1, min(k, g.size)))
    order_pred = np.argsort(-s)[:kk]
    discounts = 1.0 / np.log2(np.arange(2, kk + 2, dtype=float))
    dcg = float(np.sum(g[order_pred] * discounts))

    order_ideal = np.argsort(-g)[:kk]
    idcg = float(np.sum(g[order_ideal] * discounts))
    if idcg <= 0 or not np.isfinite(idcg):
        return 0.0
    return float(dcg / idcg)


def rank_ic_spearman(y_excess: np.ndarray, score: np.ndarray) -> float:
    y = pd.Series(np.asarray(y_excess, dtype=float))
    s = pd.Series(np.asarray(score, dtype=float))
    if len(y) < 2:
        return 0.0
    corr = y.corr(s, method="spearman")
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def daily_ranking_metrics(frame: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["as_of_date", "ndcg_at_k", "rank_ic"])

    rows: list[dict[str, float | str]] = []
    for d, g in frame.groupby("as_of_date", sort=True):
        gain = g["y_gain"].astype(float).to_numpy()
        score = g["score"].astype(float).to_numpy()
        y_excess = g["y_excess"].astype(float).to_numpy()
        rows.append(
            {
                "as_of_date": str(d),
                "ndcg_at_k": ndcg_at_k(gain, score, k=k),
                "rank_ic": rank_ic_spearman(y_excess, score),
            }
        )
    return pd.DataFrame(rows)
