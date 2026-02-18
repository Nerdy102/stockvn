from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from core.alpha_v3.calibration import build_interval_dataset, compute_interval_calibration_metrics


def test_interval_calibration_hand_example() -> None:
    golden = json.loads(Path("tests/golden/interval_calibration_hand_example.json").read_text())
    base_date = dt.date(2025, 1, 1)
    rows = []
    ys = [0.0, 0.1, 0.15, -0.1, 0.21, -0.2, 0.05, 0.0, 0.3, -0.3]
    for i in range(10):
        rows.append(
            {
                "symbol": f"S{i}",
                "as_of_date": base_date,
                "mu": 0.0,
                "interval_lo": -0.2,
                "interval_hi": 0.2 if i < 8 else 0.4,
                "bucket_id": i % 3,
            }
        )
    preds = pd.DataFrame(rows)
    labels = pd.DataFrame(
        {"symbol": [f"S{i}" for i in range(10)], "date": [base_date] * 10, "y_rank_z": ys}
    )
    feats = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(10)],
            "as_of_date": [base_date] * 10,
            "bucket_id": [i % 3 for i in range(10)],
            "regime": ["trend_up", "sideways", "risk_off", "trend_up", "sideways"] * 2,
        }
    )

    ds = build_interval_dataset(preds, labels, feats)
    metrics = compute_interval_calibration_metrics(ds, window=252, target_coverage=golden["target_coverage"])
    overall = next(x for x in metrics if x["group_key"] == "ALL")
    expected = golden["expected"]
    for k, v in expected.items():
        assert round(float(overall[k]), 6) == round(float(v), 6)
