from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlmodel import Session, select

from core.calendar_vn import get_trading_calendar_vn
from core.db.models import IndexMembership, PriceOHLCV, TickerLifecycle, UniverseAudit


class UniverseManager:
    PRESETS = {"ALL", "VN30", "VNINDEX"}

    def __init__(self, db: Session):
        self.db = db
        self.calendar = get_trading_calendar_vn()

    def universe(self, date: dt.date, name: str) -> tuple[list[str], dict[str, int]]:
        preset = str(name).upper()
        if preset not in self.PRESETS:
            raise ValueError(f"Unsupported universe preset: {name}")

        candidates = self._candidates(date, preset)
        symbols: list[str] = []
        excluded: dict[str, int] = {}

        for symbol in sorted(candidates):
            reason = self._first_exclusion_reason(symbol=symbol, date=date)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            symbols.append(symbol)

        self._upsert_audit(date=date, universe_name=preset, included_count=len(symbols), breakdown=excluded)
        return symbols, excluded

    def _candidates(self, date: dt.date, preset: str) -> set[str]:
        if preset == "ALL":
            rows = self.db.exec(select(TickerLifecycle.symbol)).all()
            return {str(symbol).upper() for symbol in rows}

        if preset in {"VNINDEX", "VN30"}:
            members = self._members_at(date=date, index_code=preset)
            if preset == "VN30" and not members:
                return self._members_at(date=date, index_code="VNINDEX")
            return members

        return set()

    def _members_at(self, date: dt.date, index_code: str) -> set[str]:
        rows = self.db.exec(
            select(IndexMembership.symbol)
            .where(IndexMembership.index_code == index_code)
            .where(IndexMembership.start_date <= date)
            .where((IndexMembership.end_date.is_(None)) | (IndexMembership.end_date >= date))
        ).all()
        return {str(symbol).upper() for symbol in rows}

    def _first_exclusion_reason(self, symbol: str, date: dt.date) -> str | None:
        lifecycle = self.db.exec(
            select(TickerLifecycle).where(TickerLifecycle.symbol == symbol)
        ).first()
        if lifecycle is None:
            return "missing_lifecycle"
        if lifecycle.first_trading_date > date:
            return "inactive_lifecycle"
        if lifecycle.last_trading_date is not None and lifecycle.last_trading_date < date:
            return "inactive_lifecycle"
        if (lifecycle.sectype or "").lower() != "stock":
            return "sectype_not_stock"

        daily = self._daily_bars(symbol=symbol, end_date=date)
        hist = daily[daily["as_of_date"] < date]
        if len(hist) < 260:
            return "insufficient_history_260"

        prev20 = self._previous_20_trading_days(hist=hist, date=date)
        if len(prev20) < 20:
            return "insufficient_adv20_window"
        adv20 = float(prev20["value_vnd"].mean())
        if adv20 < 1_000_000_000.0:
            return "adv20_below_threshold"

        on_date = daily[daily["as_of_date"] == date]
        if on_date.empty:
            return "missing_bar_on_date"
        row = on_date.iloc[-1]
        volume = float(row.get("volume", 0.0) or 0.0)
        match_value = float(row.get("value_vnd", 0.0) or 0.0)
        if volume == 0.0 and match_value == 0.0:
            return "no_trade_day"
        return None

    def _previous_20_trading_days(self, hist: pd.DataFrame, date: dt.date) -> pd.DataFrame:
        prev_date = self.calendar.prev_trading_day(date, 20)
        expected_days = self.calendar.trading_days_between(prev_date, self.calendar.prev_trading_day(date, 1), inclusive="both")
        if not expected_days:
            return pd.DataFrame(columns=["as_of_date", "value_vnd"])
        window = pd.DataFrame({"as_of_date": expected_days}).merge(
            hist[["as_of_date", "value_vnd"]], on="as_of_date", how="left"
        )
        window["value_vnd"] = window["value_vnd"].fillna(0.0)
        return window

    def _daily_bars(self, symbol: str, end_date: dt.date) -> pd.DataFrame:
        rows = self.db.exec(
            select(PriceOHLCV)
            .where(PriceOHLCV.symbol == symbol)
            .where(PriceOHLCV.timeframe == "1D")
            .where(
                PriceOHLCV.timestamp
                < dt.datetime.combine(date=end_date + dt.timedelta(days=1), time=dt.time())
            )
            .order_by(PriceOHLCV.timestamp)
        ).all()
        if not rows:
            return pd.DataFrame(columns=["as_of_date", "volume", "value_vnd"])

        frame = pd.DataFrame(
            [
                {
                    "as_of_date": r.timestamp.date(),
                    "volume": float(r.volume or 0.0),
                    "value_vnd": float(r.value_vnd or 0.0),
                }
                for r in rows
            ]
        )
        if frame.empty:
            return frame

        return frame.sort_values("as_of_date").drop_duplicates("as_of_date", keep="last")

    def _upsert_audit(
        self,
        date: dt.date,
        universe_name: str,
        included_count: int,
        breakdown: dict[str, int],
    ) -> None:
        old = self.db.exec(
            select(UniverseAudit)
            .where(UniverseAudit.date == date)
            .where(UniverseAudit.universe_name == universe_name)
        ).first()
        payload = dict(sorted(breakdown.items()))
        if old is None:
            self.db.add(
                UniverseAudit(
                    date=date,
                    universe_name=universe_name,
                    included_count=included_count,
                    excluded_json_breakdown=payload,
                )
            )
        else:
            old.included_count = included_count
            old.excluded_json_breakdown = payload
            self.db.add(old)
        self.db.commit()
