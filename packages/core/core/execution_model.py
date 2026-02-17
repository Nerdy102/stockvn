from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExecutionAssumptions:
    base_slippage_bps: float = 10.0
    k1_participation: float = 50.0
    k2_volatility: float = 40.0
    default_fill_ratio: float = 1.0
    limit_up_buy_fill_ratio: float = 0.2
    limit_down_sell_fill_ratio: float = 0.2
    participation_limit: float = 0.05


def slippage_bps(order_notional: float, adtv: float, atr_pct: float, assumptions: ExecutionAssumptions) -> float:
    a = assumptions
    part = (order_notional / adtv) if adtv > 0 else 1.0
    atrp = max(0.0, atr_pct)
    bps = a.base_slippage_bps + a.k1_participation * part + a.k2_volatility * atrp
    return float(max(0.0, bps))


def apply_slippage(price: float, side: str, slippage_bps_value: float) -> float:
    adj = slippage_bps_value / 10000.0
    if str(side).upper() == "BUY":
        return float(price * (1.0 + adj))
    return float(price * (1.0 - adj))


def execution_fill_ratio(side: str, at_upper_limit: bool, at_lower_limit: bool, assumptions: ExecutionAssumptions) -> float:
    s = str(side).upper()
    if s == "BUY" and at_upper_limit:
        return float(assumptions.limit_up_buy_fill_ratio)
    if s == "SELL" and at_lower_limit:
        return float(assumptions.limit_down_sell_fill_ratio)
    return float(assumptions.default_fill_ratio)



def load_execution_assumptions(path: str | Path) -> ExecutionAssumptions:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sl = data.get("slippage", {}) or {}
    fm = data.get("fill_model", {}) or {}
    liq = data.get("liquidity", {}) or {}
    return ExecutionAssumptions(
        base_slippage_bps=float(sl.get("base_slippage_bps", 10.0)),
        k1_participation=float(sl.get("k1_participation", 50.0)),
        k2_volatility=float(sl.get("k2_volatility", 40.0)),
        default_fill_ratio=float(fm.get("default_fill_ratio", 1.0)),
        limit_up_buy_fill_ratio=float(fm.get("limit_up_buy_fill_ratio", 0.2)),
        limit_down_sell_fill_ratio=float(fm.get("limit_down_sell_fill_ratio", 0.2)),
        participation_limit=float(liq.get("participation_limit", 0.05)),
    )
