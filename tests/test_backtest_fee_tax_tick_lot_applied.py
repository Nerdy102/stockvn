from __future__ import annotations

import datetime as dt

import pandas as pd

from api_fastapi.routers import simple_mode as sm


class _Provider:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._df.copy()


def test_backtest_fee_tax_tick_lot_applied(monkeypatch) -> None:
    dates = pd.date_range(end=dt.date.today(), periods=80, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 10123,
            "high": 10300,
            "low": 10010,
            "close": [10123 + (i % 5) * 17 for i in range(len(dates))],
            "volume": 1_000_000,
        }
    )

    def _fake_signal(model_id: str, symbol: str, timeframe: str, hist: pd.DataFrame):
        side = "BUY" if len(hist) == 21 else ("SELL" if len(hist) == 35 else "HOLD")
        return type("Sig", (), {"proposed_side": side})()

    monkeypatch.setattr(sm, "run_signal", _fake_signal)

    out = sm._run_compare_v2(
        provider=_Provider(df),
        model_id="model_1",
        symbols=["AAA"],
        timeframe="1D",
        lookback_days=252,
        execution_mode="giá đóng cửa (close)",
        include_equity_curve=True,
        include_trades=True,
    )

    assert out["trade_list"]
    trade = out["trade_list"][0]
    assert trade["qty"] % 100 == 0
    assert trade["entry_price"] % 50 == 0
    assert trade["exit_price"] % 50 == 0
    assert trade["fee"] > 0
    assert trade["tax"] > 0
