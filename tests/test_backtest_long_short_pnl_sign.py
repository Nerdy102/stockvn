from __future__ import annotations

import datetime as dt

import pandas as pd

from core.simple_mode.backtest import quick_backtest


def test_backtest_long_short_pnl_sign() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    close = [100 - i for i in range(40)]
    df = pd.DataFrame(
        {
            "date": dates.date,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000] * 40,
        }
    )
    report = quick_backtest(
        "model_1",
        "BTCUSDT",
        df,
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 9),
        position_mode="long_short",
    )
    assert isinstance(report.net_return, float)
