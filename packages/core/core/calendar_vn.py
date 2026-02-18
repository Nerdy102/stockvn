from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


class TradingCalendarVN:
    def __init__(self, config_path: str | Path = "configs/trading_calendar_vn.yaml") -> None:
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        self.timezone_name = str(data.get("timezone", "Asia/Ho_Chi_Minh"))
        self.tz_local = ZoneInfo(self.timezone_name)
        self.tz_utc = ZoneInfo("UTC")
        weekend = data.get("weekend", ["SAT", "SUN"])
        name_to_weekday = {
            "MON": 0,
            "TUE": 1,
            "WED": 2,
            "THU": 3,
            "FRI": 4,
            "SAT": 5,
            "SUN": 6,
        }
        self.weekend = {name_to_weekday[str(d).upper()] for d in weekend}
        self.holidays = {
            dt.datetime.strptime(str(d), "%Y-%m-%d").date() for d in data.get("holidays", [])
        }

    def is_trading_day(self, value: dt.date | dt.datetime) -> bool:
        d = self._as_date(value)
        return d.weekday() not in self.weekend and d not in self.holidays

    def next_trading_day(self, value: dt.date | dt.datetime, n: int = 1) -> dt.date:
        if n < 0:
            return self.prev_trading_day(value, -n)
        d = self._as_date(value)
        steps = n
        while steps > 0:
            d += dt.timedelta(days=1)
            if self.is_trading_day(d):
                steps -= 1
        return d

    def prev_trading_day(self, value: dt.date | dt.datetime, n: int = 1) -> dt.date:
        if n < 0:
            return self.next_trading_day(value, -n)
        d = self._as_date(value)
        steps = n
        while steps > 0:
            d -= dt.timedelta(days=1)
            if self.is_trading_day(d):
                steps -= 1
        return d

    def shift_trading_days(self, value: dt.date | dt.datetime, n: int) -> dt.date:
        if n == 0:
            return self._as_date(value)
        if n > 0:
            return self.next_trading_day(value, n)
        return self.prev_trading_day(value, -n)

    def trading_days_between(
        self,
        start: dt.date | dt.datetime,
        end: dt.date | dt.datetime,
        inclusive: str = "both",
    ) -> list[dt.date]:
        start_d = self._as_date(start)
        end_d = self._as_date(end)
        if end_d < start_d:
            return []

        keep_left = inclusive in {"both", "left"}
        keep_right = inclusive in {"both", "right"}
        out: list[dt.date] = []
        cur = start_d
        while cur <= end_d:
            if self.is_trading_day(cur):
                if cur == start_d and not keep_left:
                    cur += dt.timedelta(days=1)
                    continue
                if cur == end_d and not keep_right:
                    cur += dt.timedelta(days=1)
                    continue
                out.append(cur)
            cur += dt.timedelta(days=1)
        return out

    def session_windows(
        self, value: dt.date | dt.datetime
    ) -> list[tuple[dt.datetime, dt.datetime, str]]:
        d = self._as_date(value)
        if not self.is_trading_day(d):
            return []

        sessions_local = [
            (
                dt.datetime.combine(d, dt.time(hour=9, minute=0), tzinfo=self.tz_local),
                dt.datetime.combine(d, dt.time(hour=11, minute=30), tzinfo=self.tz_local),
                "morning",
            ),
            (
                dt.datetime.combine(d, dt.time(hour=13, minute=0), tzinfo=self.tz_local),
                dt.datetime.combine(d, dt.time(hour=15, minute=0), tzinfo=self.tz_local),
                "afternoon",
            ),
        ]

        out: list[tuple[dt.datetime, dt.datetime, str]] = []
        for start_local, end_local, name in sessions_local:
            start_utc = self.to_utc(start_local)
            end_utc = self.to_utc(end_local)
            out.append((self.to_local_vn(start_utc), self.to_local_vn(end_utc), name))
        return out

    def to_utc(self, dt_local_vn: dt.datetime) -> dt.datetime:
        if dt_local_vn.tzinfo is None:
            dt_local_vn = dt_local_vn.replace(tzinfo=self.tz_local)
        return dt_local_vn.astimezone(self.tz_utc)

    def to_local_vn(self, dt_utc: dt.datetime) -> dt.datetime:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=self.tz_utc)
        return dt_utc.astimezone(self.tz_local)

    @staticmethod
    def _as_date(value: dt.date | dt.datetime) -> dt.date:
        if isinstance(value, dt.datetime):
            return value.date()
        return value


@lru_cache(maxsize=1)
def get_trading_calendar_vn(config_path: str = "configs/trading_calendar_vn.yaml") -> TradingCalendarVN:
    return TradingCalendarVN(config_path=config_path)
