from __future__ import annotations

import pandas as pd

from core.monitoring.drift import compute_psi, compute_weekly_drift_metrics


def test_compute_psi_detects_shift() -> None:
    baseline = pd.Series([0.0] * 100 + [1.0] * 100)
    current = pd.Series([2.0] * 200)
    psi = compute_psi(current=current, baseline=baseline)
    assert psi > 0.25


def test_compute_weekly_drift_metrics_has_all_features() -> None:
    r = pd.Series([0.001] * 30 + [0.01] * 30)
    v = pd.Series([100] * 30 + [500] * 30)
    s = pd.Series([0.01] * 30 + [0.03] * 30)
    f = pd.Series([1.0] * 30 + [2.0] * 30)
    out = compute_weekly_drift_metrics(r, v, s, f)
    names = {m["metric_name"] for m in out}
    assert names == {"psi_returns_1d", "psi_volume", "psi_spread_proxy", "psi_flow_intensity"}
