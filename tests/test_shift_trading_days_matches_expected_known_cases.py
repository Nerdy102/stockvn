from __future__ import annotations

import datetime as dt

from core.calendar_vn import TradingCalendarVN


def test_shift_trading_days_known_tet_cases() -> None:
    cal = TradingCalendarVN("configs/trading_calendar_vn.yaml")

    assert cal.shift_trading_days(dt.date(2024, 2, 7), 1) == dt.date(2024, 2, 15)
    assert cal.shift_trading_days(dt.date(2024, 2, 15), -1) == dt.date(2024, 2, 7)
    assert cal.shift_trading_days(dt.date(2025, 1, 24), 2) == dt.date(2025, 2, 4)
