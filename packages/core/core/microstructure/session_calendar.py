from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from core.calendar_vn import get_trading_calendar_vn
from core.market_rules import MarketRules


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_utc: dt.datetime
    end_utc: dt.datetime


def build_sessions(
    value_date: dt.date,
    exchange: str,
    *,
    market_rules_path: str = "configs/market_rules_vn.yaml",
    calendar_path: str = "configs/trading_calendar_vn.yaml",
) -> list[SessionWindow]:
    cal = get_trading_calendar_vn(calendar_path)
    if not cal.is_trading_day(value_date):
        return []

    rules = MarketRules.from_yaml(market_rules_path)
    tz_local = ZoneInfo(rules.timezone)

    out: list[SessionWindow] = []
    for s in rules.sessions:
        if s.matching == "break":
            continue
        start_local = dt.datetime.combine(value_date, s.start).replace(tzinfo=tz_local)
        end_local = dt.datetime.combine(value_date, s.end).replace(tzinfo=tz_local)
        out.append(
            SessionWindow(
                name=f"{exchange}_{s.name}",
                start_utc=start_local.astimezone(dt.timezone.utc),
                end_utc=end_local.astimezone(dt.timezone.utc),
            )
        )
    return out
