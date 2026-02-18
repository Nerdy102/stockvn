from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class TradingSession:
    name: str
    start: dt.time
    end: dt.time
    matching: str
    order_types: list[str]
    notes: str = ""


@dataclass(frozen=True)
class TickRule:
    gte: float | None
    lt: float | None
    tick: int
    note: str = ""


def _normalize_exchange(v: str | None) -> str:
    return str(v or "").strip().upper()


def _normalize_instrument(v: str | None) -> str:
    s = str(v or "stock").strip().lower()
    if s in {"stock", "stocks", "fund", "funds", "stock_fund", "stock_funds"}:
        return "stock"
    if s in {"etf"}:
        return "etf"
    if s in {"cw", "covered_warrant", "covered_warrants", "warrant"}:
        return "cw"
    return s


@dataclass(frozen=True)
class MarketRules:
    """Market microstructure rules (HOSE-focused, extensible)."""

    market: str
    timezone: str

    sessions: list[TradingSession]
    put_through_sessions: list[tuple[dt.time, dt.time]]

    quantity_rules: dict[str, int]
    tick_rules_stocks: list[TickRule]
    tick_etf_cw: int
    tick_put_through: int

    price_limits: dict[str, float]
    special_cases: dict[str, Any]
    tick_rules_by_exchange: dict[str, dict[str, Any]]

    @staticmethod
    def from_yaml(path: str | Path) -> MarketRules:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        sessions: list[TradingSession] = []
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

        pts: list[tuple[dt.time, dt.time]] = []
        for s in (data.get("put_through", {}) or {}).get("sessions", []) or []:
            pts.append((_parse_time(str(s["start"])), _parse_time(str(s["end"]))))

        tick_rules: list[TickRule] = []
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

        tick_rules_by_exchange: dict[str, dict[str, Any]] = {}
        markets_cfg = (data.get("markets", {}) or {})
        if isinstance(markets_cfg, dict) and markets_cfg:
            for ex, ex_cfg in markets_cfg.items():
                ex_u = _normalize_exchange(str(ex))
                tcfg = (ex_cfg or {}).get("tick_size", {}) or {}
                stock_rules: list[TickRule] = []
                for r in (tcfg.get("stock_fund", []) or []):
                    stock_rules.append(
                        TickRule(
                            gte=float(r["gte"]) if "gte" in r and r["gte"] is not None else None,
                            lt=float(r["lt"]) if "lt" in r and r["lt"] is not None else None,
                            tick=int(r["tick"]),
                            note=str(r.get("note", "")),
                        )
                    )
                tick_rules_by_exchange[ex_u] = {
                    "stock_rules": stock_rules,
                    "etf_tick": int(((tcfg.get("etf", {}) or {}).get("tick", 10))),
                    "cw_tick": int(((tcfg.get("cw", {}) or {}).get("tick", 10))),
                    "put_through_tick": int(((tcfg.get("put_through", {}) or {}).get("tick", 1)),
                    ),
                }

        return MarketRules(
            market=str(data.get("market", "HOSE")),
            timezone=str(data.get("timezone", "Asia/Ho_Chi_Minh")),
            sessions=sessions,
            put_through_sessions=pts,
            quantity_rules={k: int(v) for k, v in (data.get("quantity_rules", {}) or {}).items()},
            tick_rules_stocks=tick_rules,
            tick_etf_cw=int(
                ((data.get("tick_size", {}) or {}).get("etf_cw", {}) or {}).get("tick", 10)
            ),
            tick_put_through=int(
                ((data.get("tick_size", {}) or {}).get("put_through", {}) or {}).get("tick", 1)
            ),
            price_limits={
                k: float(v)
                for k, v in (data.get("price_limits", {}) or {}).items()
                if k not in {"special_cases"}
            },
            special_cases=special_cases,
            tick_rules_by_exchange=tick_rules_by_exchange,
        )

    def is_trading_time(self, t: dt.time) -> bool:
        """Check if t is inside a non-break session."""
        for s in self.sessions:
            if s.matching == "break":
                continue
            if s.start <= t < s.end:
                return True
        return False

    def get_tick_size(
        self,
        price: float,
        instrument: str = "stock",
        put_through: bool = False,
        exchange: str | None = None,
    ) -> int:
        """Get tick size for given price and instrument."""
        ex = _normalize_exchange(exchange or self.market)
        inst = _normalize_instrument(instrument)

        if ex in self.tick_rules_by_exchange:
            exr = self.tick_rules_by_exchange[ex]
            if put_through:
                return int(exr.get("put_through_tick", self.tick_put_through))
            if inst == "etf":
                return int(exr.get("etf_tick", self.tick_etf_cw))
            if inst == "cw":
                return int(exr.get("cw_tick", self.tick_etf_cw))
            stock_rules = list(exr.get("stock_rules", []) or [])
        else:
            if put_through:
                return self.tick_put_through
            if inst in {"etf", "cw"}:
                return self.tick_etf_cw
            stock_rules = self.tick_rules_stocks

        for r in stock_rules:
            if r.gte is not None and price < r.gte:
                continue
            if r.lt is not None and price >= r.lt:
                continue
            return r.tick
        return 1

    def validate_tick(
        self,
        price: float,
        instrument: str = "stock",
        put_through: bool = False,
        exchange: str | None = None,
    ) -> bool:
        tick = self.get_tick_size(
            price,
            instrument=instrument,
            put_through=put_through,
            exchange=exchange,
        )
        if tick <= 0:
            return True
        # Support float input; treat values within 1e-6
        return abs((price / tick) - round(price / tick)) < 1e-6

    def round_price(
        self,
        price: float,
        instrument: str = "stock",
        put_through: bool = False,
        exchange: str | None = None,
        direction: str = "nearest",
    ) -> float:
        """Round price to tick size (nearest/up/down)."""
        tick = self.get_tick_size(
            price,
            instrument=instrument,
            put_through=put_through,
            exchange=exchange,
        )
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

    def validate_price_limit(
        self, price: float, reference_price: float, context: str = "normal"
    ) -> bool:
        """Validate price within +/- limit vs reference."""
        pct = self.price_limit_pct(context=context)
        upper = reference_price * (1.0 + pct)
        lower = reference_price * (1.0 - pct)
        return (lower - 1e-9) <= price <= (upper + 1e-9)

    def calc_price_limits(
        self, reference_price: float, context: str = "normal"
    ) -> tuple[float, float]:
        pct = self.price_limit_pct(context=context)
        return (reference_price * (1.0 - pct), reference_price * (1.0 + pct))

    def classify_session(self, t: dt.time) -> str:
        for s in self.sessions:
            if s.start <= t < s.end:
                return s.name
        return "off_session"

    def get_reference_price(
        self,
        previous_close: float,
        corporate_actions: list[dict[str, Any]] | None = None,
    ) -> float:
        """Return exchange reference price adjusted by same-day corporate actions.

        Structure-first hook: supports split and rights adjustment; cash dividend keeps previous close.
        """
        ref = float(previous_close)
        for ev in corporate_actions or []:
            action = str(ev.get("action_type", "")).upper()
            params = dict(ev.get("params_json") or ev.get("params") or {})
            if action == "SPLIT":
                factor = float(params.get("split_factor", 1.0))
                if factor > 0:
                    ref /= factor
            elif action == "RIGHTS_ISSUE":
                ratio = float(params.get("ratio", 0.0))
                sub = float(params.get("subscription_price", 0.0))
                ref = (ref + ratio * sub) / (1.0 + ratio)
            elif action == "CASH_DIVIDEND":
                cash_per_share = float(params.get("cash_per_share", 0.0))
                ref = max(0.0, ref - cash_per_share)
        return float(ref)

    def validate_order_price_qty(
        self,
        price: float,
        qty: int,
        reference_price: float,
        *,
        context: str = "normal",
        instrument: str = "stock",
        put_through: bool = False,
        allow_odd_lot: bool = False,
    ) -> bool:
        if qty <= 0:
            return False

        board_lot = int(self.quantity_rules.get("board_lot", 100))
        odd_lot_max = int(self.quantity_rules.get("odd_lot_max", 99))
        max_order = int(
            self.quantity_rules.get("max_order", self.quantity_rules.get("max_order_qty", 500000))
        )
        if qty > max_order:
            return False

        if qty < board_lot:
            if not (allow_odd_lot and qty <= odd_lot_max):
                return False
        elif qty % board_lot != 0:
            return False

        if not self.validate_tick(price, instrument=instrument, put_through=put_through):
            return False
        return self.validate_price_limit(price, reference_price=reference_price, context=context)


def _parse_time(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))


def load_market_rules(path: str | Path) -> MarketRules:
    return MarketRules.from_yaml(path)


def round_price(
    price: float,
    side: str | None = None,
    instrument_type: str = "stock",
    is_put_through: bool = False,
    *,
    path: str | Path = "configs/market_rules_vn.yaml",
    instrument: str | None = None,
    put_through: bool | None = None,
    exchange: str | None = None,
    direction: str | None = None,
) -> float:
    """Round price with side-based API and backward-compatible kwargs."""
    inst = instrument if instrument is not None else instrument_type
    pt = put_through if put_through is not None else is_put_through
    if direction is None:
        if side is None:
            direction = "nearest"
        else:
            direction = "up" if str(side).upper() == "BUY" else "down"
    return load_market_rules(path).round_price(
        price,
        instrument=inst,
        put_through=pt,
        exchange=exchange,
        direction=direction,
    )


def tick_size(
    price: float,
    instrument_type: str,
    is_put_through: bool = False,
    exchange: str | None = None,
    *,
    path: str | Path = "configs/market_rules_vn.yaml",
) -> int:
    return load_market_rules(path).get_tick_size(
        price=price,
        instrument=instrument_type,
        put_through=is_put_through,
        exchange=exchange,
    )


def clamp_qty_to_board_lot(qty: int, board_lot: int = 100) -> int:
    if qty <= 0:
        return 0
    return int(qty // board_lot * board_lot)


def round_price_by_side(
    price: float,
    side: str,
    instrument_type: str,
    is_put_through: bool = False,
    *,
    path: str | Path = "configs/market_rules_vn.yaml",
) -> int:
    direction = "up" if str(side).upper() == "BUY" else "down"
    return int(
        load_market_rules(path).round_price(
            price,
            instrument=instrument_type,
            put_through=is_put_through,
            direction=direction,
        )
    )


def calc_price_limits(
    ref_price: float,
    limit_type: str = "normal",
    *,
    path: str | Path = "configs/market_rules_vn.yaml",
) -> tuple[float, float]:
    return load_market_rules(path).calc_price_limits(reference_price=ref_price, context=limit_type)


def get_reference_price(
    previous_close: float,
    corporate_actions: list[dict[str, Any]] | None = None,
    *,
    path: str | Path = "configs/market_rules_vn.yaml",
) -> float:
    return load_market_rules(path).get_reference_price(previous_close, corporate_actions)


def validate_order(
    symbol: str,
    price: float,
    qty: int,
    session: str,
    instrument_type: str,
    *,
    reference_price: float | None = None,
    path: str | Path = "configs/market_rules_vn.yaml",
) -> list[str]:
    rules = load_market_rules(path)
    violations: list[str] = []
    if session not in {s.name for s in rules.sessions}:
        violations.append("invalid_session")
    if qty <= 0:
        violations.append("qty_non_positive")
    if qty >= 100 and qty % int(rules.quantity_rules.get("board_lot", 100)) != 0:
        violations.append("qty_not_board_lot")
    if not rules.validate_tick(price, instrument=instrument_type):
        violations.append("invalid_tick")
    ref = reference_price if reference_price is not None else price
    if not rules.validate_price_limit(price, reference_price=ref):
        violations.append("price_limit_violation")
    return violations
