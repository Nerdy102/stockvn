from __future__ import annotations

import numpy as np


def expected_cost_bps(
    target_notional: np.ndarray | float,
    adtv: np.ndarray | float,
    atr14: np.ndarray | float,
    close: np.ndarray | float,
    spread_proxy: np.ndarray | float,
) -> np.ndarray:
    target_notional_arr = np.asarray(target_notional, dtype=float)
    adtv_arr = np.asarray(adtv, dtype=float)
    atr_arr = np.asarray(atr14, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    spread_arr = np.asarray(spread_proxy, dtype=float)

    adtv_safe = np.where(adtv_arr <= 1e-12, np.nan, adtv_arr)
    close_safe = np.where(close_arr <= 1e-12, np.nan, close_arr)

    costs = (
        10.0
        + 50.0 * (target_notional_arr / adtv_safe)
        + 100.0 * (atr_arr / close_safe)
        + spread_arr * 10000.0
    )
    return np.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=1e6)


def apply_cost_penalty_to_weights(
    weights: np.ndarray,
    target_notional: np.ndarray | float,
    adtv: np.ndarray | float,
    atr14: np.ndarray | float,
    close: np.ndarray | float,
    spread_proxy: np.ndarray | float,
) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if w.sum() <= 0:
        raise ValueError("weights must have positive sum")
    w = w / w.sum()

    ecb = expected_cost_bps(
        target_notional=target_notional,
        adtv=adtv,
        atr14=atr14,
        close=close,
        spread_proxy=spread_proxy,
    )
    penalized = w / (1.0 + ecb / 50.0)
    penalized = np.maximum(penalized, 0.0)
    total = float(penalized.sum())
    if total <= 0:
        return np.ones_like(penalized) / len(penalized)
    return penalized / total
