from __future__ import annotations

import pandas as pd
from core.indicators import add_indicators


def test_add_indicators_smoke() -> None:
    df = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14] * 60,
            "high": [11, 12, 13, 14, 15] * 60,
            "low": [9, 10, 11, 12, 13] * 60,
            "close": [10, 11, 12, 13, 14] * 60,
            "volume": [100] * 300,
        }
    )
    out = add_indicators(df)
    assert "SMA20" in out.columns
    assert "EMA50" in out.columns
    assert "RSI14" in out.columns
