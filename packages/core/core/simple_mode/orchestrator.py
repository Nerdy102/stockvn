from __future__ import annotations

import datetime as dt
import uuid

from core.cost_model import SlippageConfig, calc_slippage_bps
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.simple_mode.schemas import FeeTaxPreview, OrderDraft, SignalResult


def generate_order_draft(
    *,
    signal: SignalResult,
    market_rules: MarketRules,
    fees_taxes: FeesTaxes,
    cash_available: float = 1_000_000_000.0,
    max_single_name_weight: float = 0.10,
    mode: str = "draft",
    now: dt.datetime | None = None,
    allow_short: bool = False,
    has_open_position: bool = True,
) -> OrderDraft | None:
    if signal.proposed_side == "HOLD":
        return None
    if signal.proposed_side == "SHORT" and not allow_short:
        return None
    if signal.proposed_side == "SELL" and not has_open_position:
        return None

    budget = cash_available * max_single_name_weight
    raw_qty = int(budget / max(signal.latest_price, 1.0))
    qty = max(100, (raw_qty // 100) * 100)

    rounded_price = market_rules.round_price(
        signal.latest_price,
        exchange="HOSE",
        direction="up" if signal.proposed_side == "BUY" else "down",
    )
    notional = qty * rounded_price
    slippage_bps = calc_slippage_bps(
        order_notional=notional,
        adtv=max(notional * 20, 1.0),
        atr14=max(signal.indicators.get("atr14", 0.0), 0.0),
        close=max(signal.latest_price, 1.0),
        cfg=SlippageConfig(),
    )
    slippage_est = notional * slippage_bps / 10000.0
    commission = fees_taxes.commission(notional)
    sell_tax = fees_taxes.sell_tax(notional) if signal.proposed_side == "SELL" else 0.0
    ui_side = (
        "MUA"
        if signal.proposed_side == "BUY"
        else ("MO_VI_THE_BAN" if signal.proposed_side == "SHORT" else "BAN")
    )

    _now = now or dt.datetime.now()
    off_session = not market_rules.is_trading_time(_now.time())

    return OrderDraft(
        symbol=signal.symbol,
        side=signal.proposed_side,
        ui_side=ui_side,
        qty=qty,
        price=float(rounded_price),
        notional=float(notional),
        fee_tax=FeeTaxPreview(
            commission=float(commission),
            sell_tax=float(sell_tax),
            slippage_est=float(slippage_est),
            total_cost=float(commission + sell_tax + slippage_est),
        ),
        reasons=signal.explanation,
        risks=signal.risks,
        mode=mode,
        off_session=off_session,
    )


def build_client_order_id(symbol: str) -> str:
    return f"simple-{symbol.lower()}-{uuid.uuid4().hex[:12]}"
