from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.ml.regime_v4 import (
    monthly_pit_retrain_schedule,
    monitor_regime_feature_drift,
    predict_regime_kmeans_v4,
    train_regime_kmeans_v4_pit,
)


def test_regime_v4_monthly_pit_retrain_uses_last_3y_only() -> None:
    dates = pd.date_range("2019-01-31", "2024-12-31", freq="ME")
    df = pd.DataFrame(
        {
            "as_of_date": dates,
            "f1": range(len(dates)),
            "f2": range(len(dates)),
            "f3": range(len(dates)),
            "f4": range(len(dates)),
        }
    )

    models = monthly_pit_retrain_schedule(df)
    latest = models[-1]
    assert latest.trained_on_end == pd.Timestamp("2024-12-31")
    assert latest.trained_rows == 36
    assert latest.is_fallback is False


def test_regime_v4_psi_toy_matches_golden() -> None:
    baseline = pd.DataFrame(
        {
            "f1": [0.0] * 100 + [1.0] * 100,
            "f2": [10.0] * 200,
            "f3": [5.0] * 100 + [6.0] * 100,
            "f4": [1.0] * 200,
        }
    )
    current = pd.DataFrame(
        {
            "f1": [2.0] * 200,
            "f2": [10.0] * 200,
            "f3": [5.5] * 200,
            "f4": [1.0] * 200,
        }
    )
    out = monitor_regime_feature_drift(baseline, current)
    golden_path = Path("tests/golden/regime_v4_psi_toy.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    assert out["governance_warning"] is True
    assert out["psi_threshold"] == golden["psi_threshold"]
    assert out["breached_features"] == ["f1", "f3"]
    for key, expected in golden["psi"].items():
        assert abs(float(out["psi"][key]) - float(expected)) < 1e-9


def test_regime_v4_fallback_and_prediction_works() -> None:
    tiny = pd.DataFrame(
        {
            "as_of_date": ["2024-01-31", "2024-02-29"],
            "f1": [0.1, 0.2],
            "f2": [0.1, 0.2],
            "f3": [0.1, 0.2],
            "f4": [0.1, 0.2],
        }
    )
    model = train_regime_kmeans_v4_pit(tiny, "2024-02-29")
    assert model.is_fallback is True

    pred = predict_regime_kmeans_v4(model, tiny)
    assert len(pred) == 2
    assert set(pred.tolist()) == {"sideways"}
