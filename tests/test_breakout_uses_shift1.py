from __future__ import annotations

import pandas as pd

from core.features.compute_features import compute_features


def test_breakout_uses_shift1() -> None:
    rows = []
    for i in range(40):
        rows.append({
            "date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)),
            "open": 100 + i,
            "high": 110 + i,
            "low": 90 + i,
            "close": 105 + i,
            "volume": 1000,
        })
    out = compute_features(pd.DataFrame(rows))
    expected = pd.DataFrame(rows)["high"].rolling(20).max().shift(1).iloc[-1]
    assert float(out.iloc[-1]["high20_prev"]) == float(expected)
