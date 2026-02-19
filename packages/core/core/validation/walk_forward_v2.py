from __future__ import annotations

import statistics
from typing import Any, Callable

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, run_backtest_v3

TRAIN_WINDOW = 252
TEST_WINDOW = 63
STEP = 63
MAX_FOLDS = 10


def build_walk_forward_splits(n: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    folds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    i = TRAIN_WINDOW
    while i + TEST_WINDOW <= n and len(folds) < MAX_FOLDS:
        folds.append(((i - TRAIN_WINDOW, i), (i, i + TEST_WINDOW)))
        i += STEP
    return folds


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def run_walk_forward_v2(
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: BacktestV3Config,
    signal_fn: Callable[[pd.DataFrame], str],
    fees_taxes_path: str,
    fees_crypto_path: str,
    execution_model_path: str,
) -> dict[str, Any]:
    w = df.copy().reset_index(drop=True)
    splits = build_walk_forward_splits(len(w))
    rows: list[dict[str, Any]] = []
    net_returns: list[float] = []
    mdds: list[float] = []

    for fold_id, (train_rng, test_rng) in enumerate(splits, start=1):
        train_df = w.iloc[train_rng[0] : train_rng[1]].copy()
        test_df = w.iloc[test_rng[0] : test_rng[1]].copy()
        if test_df.empty:
            continue
        baseline = {
            "liq_p20": float((train_df["close"] * train_df["volume"]).quantile(0.2)),
            "liq_p80": float((train_df["close"] * train_df["volume"]).quantile(0.8)),
            "volatility": float(train_df["close"].pct_change().std(ddof=0) or 0.0),
        }
        rep = run_backtest_v3(
            df=test_df,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            signal_fn=signal_fn,
            fees_taxes_path=fees_taxes_path,
            fees_crypto_path=fees_crypto_path,
            execution_model_path=execution_model_path,
            include_equity_curve=False,
            include_trades=False,
        )
        net_returns.append(rep.net_return)
        mdds.append(rep.mdd)
        rows.append(
            {
                "fold": fold_id,
                "train_start": train_rng[0],
                "train_end": train_rng[1],
                "test_start": test_rng[0],
                "test_end": test_rng[1],
                "dataset_hash": rep.dataset_hash,
                "baseline_stats": baseline,
                "net_return": rep.net_return,
                "mdd": rep.mdd,
                "sharpe": rep.sharpe,
                "turnover": rep.turnover,
                "costs_breakdown": rep.costs_breakdown,
            }
        )

    std_net_return = float(statistics.pstdev(net_returns)) if net_returns else 0.0
    std_mdd = float(statistics.pstdev(mdds)) if mdds else 0.0
    stability = _clamp(100.0 - (500.0 * std_net_return) - (200.0 * std_mdd), 0.0, 100.0)
    return {
        "per_fold_metrics": rows,
        "std_net_return": std_net_return,
        "std_mdd": std_mdd,
        "stability_score": stability,
    }
