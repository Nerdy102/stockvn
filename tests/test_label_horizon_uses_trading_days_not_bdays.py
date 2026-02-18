from __future__ import annotations

import datetime as dt

import pandas as pd

from core.calendar_vn import get_trading_calendar_vn
from core.ml.targets import compute_y_excess


def test_label_horizon_uses_trading_days_not_bdays() -> None:
    cal = get_trading_calendar_vn()

    dates = pd.date_range("2024-01-02", "2024-04-30", freq="D")
    rows: list[dict] = []
    for sym, base in [("AAA", 100.0), ("VNINDEX", 1000.0)]:
        px = base
        for ts in dates:
            d = ts.date()
            if cal.is_trading_day(d):
                px += 1.0
                rows.append({"symbol": sym, "timestamp": ts, "close": px})

    out = compute_y_excess(pd.DataFrame(rows), horizon=21)
    one = out[(out["symbol"] == "AAA") & (out["as_of_date"] == dt.date(2024, 1, 8))].iloc[0]
    assert one["realized_date"] == cal.shift_trading_days(dt.date(2024, 1, 8), 21)
