from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class TradingSession:
    name: str
    start: dt.time
    end: dt.time
    matching: str
    order_types: List[str]
    notes: str = ""


@dataclass(frozen=True)
class TickRule:
    gte: Optional[float]
    lt: Optional[float]
    tick: int
    note: str = ""


@dataclass(frozen=True)
class MarketRules:
    """Market microstructure rules (HOSE-focused, extensible)."""

    market: str
    timezone: str

    sessions: List[TradingSession]
    put_through_sessions: List[Tuple[dt.time, dt.time]]

    quantity_rules: Dict[str, int]
    tick_rules_stocks: List[TickRule]
    tick_etf_cw: int
    tick_put_through: int

    price_limits: Dict[str, float]
    special_cases: Dict[str, Any]

    @staticmethod
    def from_yaml(path: str | Path) -> "MarketRules":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        sessions: List[TradingSession] = []
        for s in data.get("trading_hours", []) or []:
            sessions.append(
                TradingSession(
                    name=str(s["name"]),
                    start=_parse_time(str(s["start"])),
                    end=_parse_time(str(s["end"])),
                    matching=str(s.get("matching", "")),
                    order_types=list(s.get("order_types", []) or []),
                    notes=str(s.get("notes", "")),
                )
            )

        pts: List[Tuple[dt.time, dt.time]] = []
        for s in (data.get("put_through", {}) or {}).get("sessions", []) or []:
            pts.append((_parse_time(str(s["start"])), _parse_time(str(s["end"]))))

        tick_rules: List[TickRule] = []
        for r in (data.get("tick_size", {}) or {}).get("stocks_funds", []) or []:
            tick_rules.append(
                TickRule(
                    gte=float(r["gte"]) if "gte" in r and r["gte"] is not None else None,
                    lt=float(r["lt"]) if "lt" in r and r["lt"] is not None else None,
                    tick=int(r["tick"]),
                    note=str(r.get("note", "")),
                )
            )

        special_cases = dict((data.get("price_limits", {}) or {}).get("special_cases", {}) or {})

        return MarketRules(
            market=str(data.get("market", "HOSE")),
            timezone=str(data.get("timezone", "Asia/Ho_Chi_Minh")),
            sessions=sessions,
            put_through_sessions=pts,
            quantity_rules={k: int(v) for k, v in (data.get("quantity_rules", {}) or {}).items()},
            tick_rules_stocks=tick_rules,
            tick_etf_cw=int(((data.get("tick_size", {}) or {}).get("etf_cw", {}) or {}).get("tick", 10)),
            tick_put_through=int(((data.get("tick_size", {}) or {}).get("put_through", {}) or {}).get("tick", 1)),
            price_limits={
                k: float(v)
                for k, v in (data.get("price_limits", {}) or {}).items()
                if k not in {"special_cases"}
            },
            special_cases=special_cases,
        )

    def is_trading_time(self, t: dt.time) -> bool:
        """Check if t is inside a non-break session."""
        for s in self.sessions:
            if s.matching == "break":
                continue
            if s.start <= t < s.end:
                return True
        return False

    def get_tick_size(self, price: float, instrument: str = "stock", put_through: bool = False) -> int:
        """Get tick size for given price and instrument."""
        if put_through:
            return self.tick_put_through
        if instrument.lower() in {"etf", "cw"}:
            return self.tick_etf_cw
        for r in self.tick_rules_stocks:
            if r.gte is not None and price < r.gte:
                continue
            if r.lt is not None and price >= r.lt:
                continue
            return r.tick
        return 1

    def validate_tick(self, price: float, instrument: str = "stock", put_through: bool = False) -> bool:
        tick = self.get_tick_size(price, instrument=instrument, put_through=put_through)
        if tick <= 0:
            return True
        # Support float input; treat values within 1e-6
        return abs((price / tick) - round(price / tick)) < 1e-6

    def round_price(
        self,
        price: float,
        instrument: str = "stock",
        put_through: bool = False,
        direction: str = "nearest",
    ) -> float:
        """Round price to tick size (nearest/up/down)."""
        tick = self.get_tick_size(price, instrument=instrument, put_through=put_through)
        if tick <= 0:
            return price
        x = price / tick
        if direction == "down":
            return int(x) * tick
        if direction == "up":
            return (int(x) + (0 if float(x).is_integer() else 1)) * tick
        return round(x) * tick

    def price_limit_pct(self, context: str = "normal") -> float:
        """Return price limit percent for a context key.

        context in: normal | first_trading_day | resumed_after_suspension_25d | ...
        special cases are stored in self.special_cases for extension (e.g., ex-rights).
        """
        if context in self.price_limits:
            return float(self.price_limits[context])
        return float(self.price_limits.get("normal", 0.07))

    def validate_price_limit(self, price: float, reference_price: float, context: str = "normal") -> bool:
        """Validate price within +/- limit vs reference."""
        pct = self.price_limit_pct(context=context)
        upper = reference_price * (1.0 + pct)
        lower = reference_price * (1.0 - pct)
        return (lower - 1e-9) <= price <= (upper + 1e-9)


def _parse_time(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))


def load_market_rules(path: str | Path) -> MarketRules:
    return MarketRules.from_yaml(path)
