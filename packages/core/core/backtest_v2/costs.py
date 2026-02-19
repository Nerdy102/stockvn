from __future__ import annotations

import math


def slippage_bps(
    base_bps: float, k_atr: float, k_liq: float, atr_pct: float, dollar_volume: float
) -> float:
    norm_liq = max(dollar_volume, 1.0)
    return base_bps + k_atr * (atr_pct * 100.0) + k_liq / math.sqrt(norm_liq)


def apply_price_with_slippage(price: float, side: str, bps: float) -> float:
    if side in {"BUY", "LONG"}:
        return price * (1.0 + bps / 10_000.0)
    return price * (1.0 - bps / 10_000.0)
