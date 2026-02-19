from __future__ import annotations

import math


def round_down_to_board_lot(qty_raw: float, lot: int = 100) -> int:
    if lot <= 0:
        return 0
    return int(qty_raw // lot) * lot


def round_down_to_step(qty_raw: float, step_size: float = 0.0001) -> float:
    if step_size <= 0:
        step_size = 0.0001
    return math.floor(qty_raw / step_size) * step_size


def compute_position_size(
    *,
    nav: float,
    close: float,
    atr14: float,
    market: str,
    target_risk_per_trade_pct: float = 0.005,
    max_position_notional_pct: float = 0.2,
    min_cash_buffer_pct: float = 0.05,
    crypto_step_size: float = 0.0001,
) -> tuple[float, str | None]:
    stop_distance = max(2 * atr14, 0.02 * close)
    qty_raw = (nav * target_risk_per_trade_pct) / max(stop_distance, 1e-9)
    qty = (
        float(round_down_to_board_lot(qty_raw, lot=100))
        if market == "vn"
        else float(round_down_to_step(qty_raw, step_size=crypto_step_size))
    )
    notional = qty * close
    if notional > nav * max_position_notional_pct or (nav - notional) < nav * min_cash_buffer_pct:
        return 0.0, "Vượt giới hạn rủi ro."
    return qty, None
