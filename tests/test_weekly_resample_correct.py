from __future__ import annotations

import pandas as pd

from services.api_fastapi.api_fastapi.routers.chart import _resample_weekly


def test_weekly_resample_correct() -> None:
    ts = pd.date_range("2025-01-06", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "high": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "low": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "volume": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "value_vnd": [100] * 10,
        }
    )
    out = _resample_weekly(df)
    first = out.iloc[0]
    assert float(first["open"]) == 1.0
    assert float(first["high"]) == 6.0
    assert float(first["low"]) == 0.0
    assert float(first["close"]) == 5.5
    assert float(first["volume"]) == 150.0
