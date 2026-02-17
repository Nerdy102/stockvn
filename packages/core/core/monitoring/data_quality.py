from __future__ import annotations

from typing import Any

import pandas as pd


def compute_data_quality_metrics(
    df: pd.DataFrame, provider: str, timeframe: str
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if df.empty:
        return metrics

    d = df.sort_values(["symbol", "timestamp"]).copy()

    # monotonic timestamp per symbol
    for sym, g in d.groupby("symbol"):
        monotonic = bool(pd.Series(g["timestamp"]).is_monotonic_increasing)
        metrics.append(
            {
                "provider": provider,
                "symbol": sym,
                "timeframe": timeframe,
                "metric_name": "monotonic_ts",
                "metric_value": 1.0 if monotonic else 0.0,
            }
        )

    # OHLC sanity
    ohlc_ok = (
        (d["high"] >= d[["open", "close"]].max(axis=1))
        & (d["low"] <= d[["open", "close"]].min(axis=1))
        & (d["volume"] >= 0)
    ).mean()
    metrics.append(
        {
            "provider": provider,
            "symbol": None,
            "timeframe": timeframe,
            "metric_name": "ohlc_sanity_ratio",
            "metric_value": float(ohlc_ok),
        }
    )

    # duplicate ratio
    dup_ratio = float(d.duplicated(subset=["symbol", "timestamp", "timeframe"]).mean())
    metrics.append(
        {
            "provider": provider,
            "symbol": None,
            "timeframe": timeframe,
            "metric_name": "duplicate_ratio",
            "metric_value": dup_ratio,
        }
    )

    # missing ratio by field
    for col in ["open", "high", "low", "close", "volume", "value_vnd"]:
        if col in d.columns:
            metrics.append(
                {
                    "provider": provider,
                    "symbol": None,
                    "timeframe": timeframe,
                    "metric_name": f"missing_ratio_{col}",
                    "metric_value": float(d[col].isna().mean()),
                }
            )
    return metrics
