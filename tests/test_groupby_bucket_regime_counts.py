from __future__ import annotations

import datetime as dt

import pandas as pd

from core.alpha_v3.calibration import build_interval_dataset, compute_interval_calibration_metrics


def test_groupby_bucket_regime_counts() -> None:
    d = dt.date(2025, 1, 3)
    preds = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "as_of_date": [d, d, d, d],
            "interval_lo": [-1, -1, -1, -1],
            "interval_hi": [1, 1, 1, 1],
            "bucket_id": [0, 1, 2, 2],
        }
    )
    labels = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "date": [d] * 4, "y_rank_z": [0.0, 0.0, 0.0, 0.0]})
    feats = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "as_of_date": [d] * 4,
            "bucket_id": [0, 1, 2, 2],
            "regime": ["trend_up", "sideways", "risk_off", "risk_off"],
        }
    )

    ds = build_interval_dataset(preds, labels, feats)
    out = compute_interval_calibration_metrics(ds)
    counts = {r["group_key"]: r["count"] for r in out}
    assert counts["ALL"] == 4
    assert counts["bucket:LOW"] == 1
    assert counts["bucket:MID"] == 1
    assert counts["bucket:HIGH"] == 2
    assert counts["bucket:HIGH|regime:RISK_OFF"] == 2
