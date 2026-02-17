from __future__ import annotations

from typing import Any

import pandas as pd


def compute_drift_zscore(current: pd.Series, baseline: pd.Series) -> float:
    if baseline.std() == 0 or len(current) == 0:
        return 0.0
    return float((current.mean() - baseline.mean()) / baseline.std())


def compute_weekly_drift_metrics(
    returns: pd.Series, volume: pd.Series, spread_proxy: pd.Series
) -> list[dict[str, Any]]:
    n = len(returns)
    if n < 20:
        return []
    split = max(10, n // 2)
    base_r, cur_r = returns.iloc[:split], returns.iloc[split:]
    base_v, cur_v = volume.iloc[:split], volume.iloc[split:]
    base_s, cur_s = spread_proxy.iloc[:split], spread_proxy.iloc[split:]

    return [
        {
            "metric_name": "drift_returns_z",
            "metric_value": compute_drift_zscore(cur_r, base_r),
            "alert": abs(compute_drift_zscore(cur_r, base_r)) > 3,
        },
        {
            "metric_name": "drift_volume_z",
            "metric_value": compute_drift_zscore(cur_v, base_v),
            "alert": abs(compute_drift_zscore(cur_v, base_v)) > 3,
        },
        {
            "metric_name": "drift_spread_proxy_z",
            "metric_value": compute_drift_zscore(cur_s, base_s),
            "alert": abs(compute_drift_zscore(cur_s, base_s)) > 3,
        },
    ]
