from __future__ import annotations

import datetime as dt

from core.calendar_vn import get_trading_calendar_vn
from data.providers.csv_provider import CsvProvider


def test_intraday_seed_anchor_uses_last_daily_date() -> None:
    provider = CsvProvider("data_demo")
    symbols = provider.get_tickers()["symbol"].tolist()

    prefer = ["VNINDEX"] + [s for s in symbols if s != "VNINDEX"]
    last_date = None
    for sym in prefer:
        daily = provider.get_ohlcv(sym, "1D")
        if daily is not None and not daily.empty:
            last_date = daily["date"].max()
            break

    assert last_date is not None
    cal = get_trading_calendar_vn()
    start = cal.shift_trading_days(last_date, -20)
    assert start < last_date
    assert last_date != dt.date.today()
