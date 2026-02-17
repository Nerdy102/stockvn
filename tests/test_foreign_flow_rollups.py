from __future__ import annotations

import pandas as pd

from core.ml.features_v2 import compute_foreign_flow_features


def test_foreign_flow_rollups() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    meta = pd.DataFrame(
        {
            "symbol": ["AAA"] * 25,
            "as_of_date": dates.date,
            "net_foreign_val": [1.0] * 25,
            "current_room": [40.0] * 25,
            "total_room": [100.0] * 25,
        }
    )
    adv = pd.DataFrame({"symbol": ["AAA"] * 25, "as_of_date": dates.date, "adv20_value": [10.0] * 25})
    out = compute_foreign_flow_features(meta, adv)
    last = out.iloc[-1]
    assert abs(last["net_foreign_val_5d"] - 5.0) < 1e-9
    assert abs(last["net_foreign_val_20d"] - 20.0) < 1e-9
