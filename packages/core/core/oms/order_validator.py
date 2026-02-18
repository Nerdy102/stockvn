from __future__ import annotations

import datetime as dt
from typing import Any

from sqlmodel import Session, select

from core.db.models import PriceOHLCV, Ticker, Trade
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules

MAX_SINGLE_NAV = 0.10
MAX_SECTOR_NAV = 0.25
MIN_CASH_NAV = 0.05
LIQUIDITY_CAP_MULTIPLIER = 0.05 * 3.0


def _latest_marks(session: Session) -> dict[str, float]:
    rows = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    marks: dict[str, tuple[dt.datetime, float]] = {}
    for r in rows:
        cur = marks.get(r.symbol)
        if cur is None or r.timestamp > cur[0]:
            marks[r.symbol] = (r.timestamp, float(r.close))
    return {k: v[1] for k, v in marks.items()}


def _avg_adtv(session: Session, symbol: str, lookback: int = 20) -> float:
    rows = session.exec(
        select(PriceOHLCV)
        .where(PriceOHLCV.symbol == symbol)
        .where(PriceOHLCV.timeframe == "1D")
        .order_by(PriceOHLCV.timestamp.desc())
        .limit(lookback)
    ).all()
    if not rows:
        return 0.0
    vals = [float(r.value_vnd) if float(r.value_vnd) > 0 else float(r.close) * float(r.volume) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _portfolio_snapshot(session: Session, portfolio_id: int, fees: FeesTaxes, broker_name: str) -> dict[str, Any]:
    trades = session.exec(select(Trade).where(Trade.portfolio_id == portfolio_id)).all()
    marks = _latest_marks(session)

    qty: dict[str, float] = {}
    cash = 1_000_000_000.0
    for t in sorted(trades, key=lambda x: (x.trade_date, x.id or 0)):
        side = str(t.side).upper()
        q = float(t.quantity)
        px = float(t.price)
        notional = q * px
        comm = fees.commission(notional, broker_name)
        tax = fees.sell_tax(notional) if side == "SELL" else 0.0
        if side == "BUY":
            qty[t.symbol] = qty.get(t.symbol, 0.0) + q
            cash -= notional + comm + tax
        else:
            current = qty.get(t.symbol, 0.0)
            exec_q = min(q, current)  # SELL clamp bugfix remains
            qty[t.symbol] = current - exec_q
            cash += exec_q * px - comm - tax

    qty = {k: v for k, v in qty.items() if v > 1e-9}
    holdings_value = sum(float(v) * float(marks.get(k, 0.0)) for k, v in qty.items())
    nav = cash + holdings_value
    return {"qty": qty, "cash": cash, "marks": marks, "nav": nav}


def validate_pretrade(
    *,
    session: Session,
    portfolio_id: int,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    as_of: dt.datetime,
    market_rules: MarketRules,
    fees: FeesTaxes,
    broker_name: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    side_u = str(side).upper()
    qty = float(quantity)
    px = float(price)

    if side_u not in {"BUY", "SELL"}:
        reasons.append("invalid_side")
    if qty <= 0 or px <= 0:
        reasons.append("invalid_qty_or_price")

    session_name = market_rules.classify_session(as_of.time())
    if session_name == "off_session" or session_name.lower() == "break":
        reasons.append("invalid_session")

    if not market_rules.validate_tick(px):
        reasons.append("invalid_tick")

    board_lot = int(market_rules.quantity_rules.get("board_lot", 100))
    if board_lot > 0 and (qty % board_lot) != 0:
        reasons.append("invalid_lot")

    snap = _portfolio_snapshot(session, portfolio_id, fees, broker_name)
    nav = float(snap["nav"])
    marks = dict(snap["marks"])
    marks[symbol] = px
    qty_map = dict(snap["qty"])

    if side_u == "BUY":
        qty_map[symbol] = qty_map.get(symbol, 0.0) + qty
    else:
        qty_map[symbol] = max(0.0, qty_map.get(symbol, 0.0) - qty)

    position_value = float(qty_map.get(symbol, 0.0)) * px
    if nav > 0 and position_value > MAX_SINGLE_NAV * nav + 1e-9:
        reasons.append("max_single_nav_breach")

    tickers = session.exec(select(Ticker)).all()
    sector_by_symbol = {t.symbol: str(t.sector or "Unknown") for t in tickers}
    sector = sector_by_symbol.get(symbol, "Unknown")
    sector_value = 0.0
    for sym, qv in qty_map.items():
        if sector_by_symbol.get(sym, "Unknown") == sector:
            sector_value += float(qv) * float(marks.get(sym, 0.0))
    if nav > 0 and sector_value > MAX_SECTOR_NAV * nav + 1e-9:
        reasons.append("max_sector_nav_breach")

    notional = qty * px
    comm = fees.commission(notional, broker_name)
    tax = fees.sell_tax(notional) if side_u == "SELL" else 0.0
    cash_after = float(snap["cash"])
    if side_u == "BUY":
        cash_after -= notional + comm + tax
    else:
        cash_after += notional - comm - tax
    if nav > 0 and cash_after < MIN_CASH_NAV * nav - 1e-9:
        reasons.append("min_cash_nav_breach")

    adtv = _avg_adtv(session, symbol)
    liquidity_cap = adtv * LIQUIDITY_CAP_MULTIPLIER
    if liquidity_cap > 0 and position_value > liquidity_cap + 1e-9:
        reasons.append("liquidity_cap_breach")

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "nav": nav,
            "cash_after": cash_after,
            "position_value": position_value,
            "sector_value": sector_value,
            "adtv": adtv,
            "liquidity_cap": liquidity_cap,
            "session": session_name,
        },
    }
