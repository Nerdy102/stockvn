from __future__ import annotations

import datetime as dt

from core.calendar_vn import TradingCalendarVN


def test_calendar_skips_weekends_and_holidays() -> None:
    cal = TradingCalendarVN("configs/trading_calendar_vn.yaml")

    assert cal.is_trading_day(dt.date(2024, 2, 6))
    assert not cal.is_trading_day(dt.date(2024, 2, 10))  # Saturday
    assert not cal.is_trading_day(dt.date(2024, 2, 12))  # Tet holiday

    got = cal.trading_days_between(dt.date(2024, 2, 8), dt.date(2024, 2, 16), inclusive="both")
    assert got == [dt.date(2024, 2, 15), dt.date(2024, 2, 16)]
