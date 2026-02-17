from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageConfig:
    base_bps: float = 10.0
    k1: float = 50.0
    k2: float = 100.0


@dataclass(frozen=True)
class FillConfig:
    market_fill_ratio: float = 1.0
    limit_fill_ratio_if_crossed: float = 1.0
    limit_fill_ratio_if_not_crossed: float = 0.0
    limit_up_buy_penalty: float = 0.2
    limit_down_sell_penalty: float = 0.2


def calc_slippage_bps(
    order_notional: float, adtv: float, atr14: float, close: float, cfg: SlippageConfig
) -> float:
    participation = (order_notional / adtv) if adtv > 0 else 1.0
    atr_ratio = (atr14 / close) if close > 0 else 0.0
    bps = cfg.base_bps + cfg.k1 * max(0.0, participation) + cfg.k2 * max(0.0, atr_ratio)
    return float(max(0.0, bps))


def apply_execution_slippage(price: float, side: str, slippage_bps: float) -> float:
    m = slippage_bps / 10000.0
    if side.upper() == "BUY":
        return float(price * (1.0 + m))
    return float(price * (1.0 - m))


def calc_fill_ratio(
    side: str,
    order_type: str,
    *,
    crossed: bool = True,
    at_upper_limit: bool = False,
    at_lower_limit: bool = False,
    cfg: FillConfig = FillConfig(),
) -> float:
    s = side.upper()
    ot = order_type.upper()
    if ot == "MARKET":
        ratio = cfg.market_fill_ratio
    else:
        ratio = cfg.limit_fill_ratio_if_crossed if crossed else cfg.limit_fill_ratio_if_not_crossed

    if s == "BUY" and at_upper_limit:
        ratio *= cfg.limit_up_buy_penalty
    if s == "SELL" and at_lower_limit:
        ratio *= cfg.limit_down_sell_penalty
    return float(min(1.0, max(0.0, ratio)))
