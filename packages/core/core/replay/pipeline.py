from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules


@dataclass
class PositionState:
    qty: int = 0
    avg_cost: float = 0.0


@dataclass
class ReplayResult:
    positions: dict[str, int]
    cash: float
    fees_total: float
    taxes_total: float
    realized_pnl: float
    position_eod: dict[str, dict[str, int]]
    cash_ledger: dict[str, float]


class UnifiedTradingPipeline:
    """Single execution path shared by backtest/replay+paper."""

    def __init__(
        self,
        *,
        market_rules: MarketRules,
        fees_taxes: FeesTaxes,
        cost_model: Callable[[float], float],
        corporate_action_ledger: Callable[[dict[str, Any], dict[str, PositionState]], None],
        calendar_vn: Any,
    ) -> None:
        self.market_rules = market_rules
        self.fees_taxes = fees_taxes
        self.cost_model = cost_model
        self.corporate_action_ledger = corporate_action_ledger
        self.calendar_vn = calendar_vn

    def run(self, events: list[dict[str, Any]], *, initial_cash: float) -> ReplayResult:
        positions: dict[str, PositionState] = defaultdict(PositionState)
        pending_signal: dict[str, int] = {}
        cash = float(initial_cash)
        fees_total = 0.0
        taxes_total = 0.0
        realized_pnl = 0.0
        position_eod: dict[str, dict[str, int]] = {}
        cash_ledger: dict[str, float] = {}

        ordered_events = sorted(events, key=lambda x: (x["ts_utc"], x.get("id", 0)))
        for event in ordered_events:
            ts = _to_dt(event["ts_utc"])
            d = ts.date()
            if not self.calendar_vn.is_trading_day(d):
                continue
            self.corporate_action_ledger(event, positions)

            et = event["event_type"]
            payload = event["payload_json"]
            symbol = event.get("symbol") or payload.get("symbol")
            if not symbol:
                day_key = d.isoformat()
                position_eod[day_key] = {
                    sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
                }
                cash_ledger[day_key] = float(cash)
                continue

            if et == "signal":
                pending_signal[symbol] = int(payload.get("target", 0))
                day_key = d.isoformat()
                position_eod[day_key] = {
                    sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
                }
                cash_ledger[day_key] = float(cash)
                continue

            if et != "bar" or symbol not in pending_signal:
                day_key = d.isoformat()
                position_eod[day_key] = {
                    sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
                }
                cash_ledger[day_key] = float(cash)
                continue

            target = pending_signal.pop(symbol)
            px = float(payload.get("open", payload.get("price", 0.0)) or 0.0)
            if px <= 0:
                day_key = d.isoformat()
                position_eod[day_key] = {
                    sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
                }
                cash_ledger[day_key] = float(cash)
                continue

            state = positions[symbol]
            side = None
            order_qty = 0
            if target > 0 and state.qty == 0:
                side = "BUY"
                order_qty = 100
            elif target <= 0 and state.qty > 0:
                side = "SELL"
                order_qty = state.qty

            if not side or order_qty <= 0:
                day_key = d.isoformat()
                position_eod[day_key] = {
                    sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
                }
                cash_ledger[day_key] = float(cash)
                continue

            slipped_px = px * (1.0 + self.cost_model(px) if side == "BUY" else 1.0 - self.cost_model(px))
            exec_px = self.market_rules.round_price(slipped_px, direction=("up" if side == "BUY" else "down"))
            notional = exec_px * order_qty
            fee = self.fees_taxes.commission(notional)
            tax = self.fees_taxes.sell_tax(notional) if side == "SELL" else 0.0

            if side == "BUY":
                if cash < notional + fee:
                    day_key = d.isoformat()
                    position_eod[day_key] = {
                        sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
                    }
                    cash_ledger[day_key] = float(cash)
                    continue
                new_qty = state.qty + order_qty
                state.avg_cost = (state.avg_cost * state.qty + notional + fee) / max(new_qty, 1)
                state.qty = new_qty
                cash -= notional + fee
            else:
                pnl = (exec_px - state.avg_cost) * order_qty - fee - tax
                realized_pnl += pnl
                cash += notional - fee - tax
                state.qty -= order_qty
                if state.qty == 0:
                    state.avg_cost = 0.0
            fees_total += fee
            taxes_total += tax

            day_key = d.isoformat()
            position_eod[day_key] = {
                sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0
            }
            cash_ledger[day_key] = float(cash)

        if ordered_events:
            last_day = _to_dt(ordered_events[-1]["ts_utc"]).date().isoformat()
            position_eod.setdefault(last_day, {sym: pos.qty for sym, pos in sorted(positions.items()) if pos.qty != 0})
            cash_ledger.setdefault(last_day, float(cash))

        return ReplayResult(
            positions={k: v.qty for k, v in positions.items() if v.qty != 0},
            cash=float(cash),
            fees_total=float(fees_total),
            taxes_total=float(taxes_total),
            realized_pnl=float(realized_pnl),
            position_eod=position_eod,
            cash_ledger=cash_ledger,
        )


def _to_dt(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
