from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_psi(current: pd.Series, baseline: pd.Series, bins: int = 10) -> float:
    cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)
    base = pd.to_numeric(baseline, errors="coerce").dropna().astype(float)
    if cur.empty or base.empty:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    cuts = np.unique(np.quantile(base, quantiles))
    if len(cuts) <= 2:
        return 0.0

    base_hist, _ = np.histogram(base, bins=cuts)
    cur_hist, _ = np.histogram(cur, bins=cuts)

    base_pct = np.clip(base_hist / max(base_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    psi = np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct))
    return float(psi)


def compute_weekly_drift_metrics(
    returns_1d: pd.Series,
    volume: pd.Series,
    spread_proxy: pd.Series,
    flow_intensity: pd.Series,
) -> list[dict[str, Any]]:
    n = min(len(returns_1d), len(volume), len(spread_proxy), len(flow_intensity))
    if n < 20:
        return []
    split = max(10, n // 2)

    series = {
        "returns_1d": returns_1d.iloc[:n],
        "volume": volume.iloc[:n],
        "spread_proxy": spread_proxy.iloc[:n],
        "flow_intensity": flow_intensity.iloc[:n],
    }

    out: list[dict[str, Any]] = []
    for metric_name, s in series.items():
        baseline = s.iloc[:split]
        current = s.iloc[split:]
        psi = compute_psi(current=current, baseline=baseline)
        out.append(
            {
                "metric_name": f"psi_{metric_name}",
                "metric_value": float(psi),
                "alert": bool(psi > 0.25),
            }
        )
    return out
