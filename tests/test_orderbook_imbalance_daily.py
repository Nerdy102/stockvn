from __future__ import annotations

import pandas as pd

from core.ml.features_v2 import compute_orderbook_daily_features


def test_orderbook_imbalance_daily() -> None:
    q = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-01-01 09:00:00", "bidVol1": 100, "askVol1": 50, "bidVol2": 80, "askVol2": 60, "bidVol3": 70, "askVol3": 65, "bidPrice1": 10.0, "askPrice1": 10.2},
            {"symbol": "AAA", "timestamp": "2024-01-01 09:05:00", "bidVol1": 90, "askVol1": 60, "bidVol2": 85, "askVol2": 70, "bidVol3": 75, "askVol3": 60, "bidPrice1": 10.0, "askPrice1": 10.1},
        ]
    )
    out = compute_orderbook_daily_features(q)
    assert {"imb_1_day", "imb_3_day", "spread_day"}.issubset(set(out.columns))
    assert len(out) == 1
