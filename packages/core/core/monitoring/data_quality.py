from __future__ import annotations

from typing import Any

import pandas as pd


def compute_data_quality_metrics(
    df: pd.DataFrame, provider: str, timeframe: str
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if df.empty:
        return metrics

    d = df.copy()
    total = max(len(d), 1)

    dup_rate = float(d.duplicated(subset=["symbol", "timestamp", "timeframe"]).mean())
    metrics.append(
        {
            "provider": provider,
            "symbol": None,
            "timeframe": timeframe,
            "metric_name": "duplicate_rate",
            "metric_value": dup_rate,
        }
    )

    for col in ["open", "high", "low", "close", "volume", "value_vnd"]:
        if col in d.columns:
            metrics.append(
                {
                    "provider": provider,
                    "symbol": None,
                    "timeframe": timeframe,
                    "metric_name": f"missing_rate_{col}",
                    "metric_value": float(d[col].isna().sum() / total),
                }
            )

    required = {"open", "high", "low", "close", "volume"}
    if required.issubset(set(d.columns)):
        violate = (
            (d["high"] < d[["open", "close"]].max(axis=1))
            | (d["low"] > d[["open", "close"]].min(axis=1))
            | (d["high"] < d["low"])
            | (d["volume"] < 0)
        )
        metrics.append(
            {
                "provider": provider,
                "symbol": None,
                "timeframe": timeframe,
                "metric_name": "ohlc_invariants_violation_rate",
                "metric_value": float(violate.mean()),
            }
        )

    return metrics
