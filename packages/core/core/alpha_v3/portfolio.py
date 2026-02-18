from __future__ import annotations

from typing import Any

import numpy as np

from core.market_rules import clamp_qty_to_board_lot, round_price

from core.alpha_v3.costs import apply_cost_penalty_to_weights
from core.alpha_v3.hrp import compute_hrp_weights




def _validate_same_length(n: int, **arrays: np.ndarray) -> None:
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"{name} length {len(arr)} does not match expected {n}")

def _normalize_positive(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _redistribute_with_upper_bounds(
    base_weights: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    target_sum: float,
) -> np.ndarray:
    """Project positive weights to an upper-bounded simplex.

    Returns vector x such that:
    - 0 <= x_i <= upper_bounds_i
    - sum(x) ~= min(target_sum, sum(upper_bounds))
    """
    w = _normalize_positive(base_weights)
    ub = np.maximum(np.asarray(upper_bounds, dtype=float), 0.0)
    feasible_total = min(float(target_sum), float(ub.sum()))
    if feasible_total <= 0:
        return np.zeros_like(w)

    out = np.minimum(w * feasible_total, ub)

    for _ in range(10 * len(w) + 20):
        deficit = feasible_total - float(out.sum())
        if deficit <= 1e-12:
            break
        room = ub - out
        eligible = room > 1e-12
        if not np.any(eligible):
            break

        mass = float(w[eligible].sum())
        if mass <= 1e-12:
            add = np.zeros_like(out)
            add[eligible] = deficit / int(np.sum(eligible))
        else:
            add = np.zeros_like(out)
            add[eligible] = deficit * (w[eligible] / mass)

        out = np.minimum(out + add, ub)

    return np.maximum(out, 0.0)


def _apply_sector_caps(
    base_weights: np.ndarray,
    sectors: np.ndarray,
    *,
    sector_cap: float,
    upper_bounds: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    out = np.maximum(np.asarray(base_weights, dtype=float), 0.0)
    ub = np.maximum(np.asarray(upper_bounds, dtype=float), 0.0)
    secs = np.asarray(sectors, dtype=object)
    target = min(float(target_sum), float(ub.sum()))

    for _ in range(20 * len(out) + 20):
        # 1) Enforce sector caps by proportional shrink inside violating sectors
        sector_sums: dict[str, float] = {}
        for i, s in enumerate(secs):
            k = str(s)
            sector_sums[k] = sector_sums.get(k, 0.0) + float(out[i])

        violating = {k: v for k, v in sector_sums.items() if v > sector_cap + 1e-12}
        if violating:
            for k, total in violating.items():
                idx = np.where(secs.astype(str) == k)[0]
                out[idx] *= sector_cap / total

        # 2) Redistribute leftover to names with both name-room and sector-room
        deficit = target - float(out.sum())
        if deficit <= 1e-12:
            break

        sector_sums = {}
        for i, s in enumerate(secs):
            k = str(s)
            sector_sums[k] = sector_sums.get(k, 0.0) + float(out[i])

        asset_room = np.maximum(0.0, ub - out)
        sec_room = np.array(
            [max(0.0, sector_cap - sector_sums.get(str(secs[i]), 0.0)) for i in range(len(out))],
            dtype=float,
        )
        room = np.minimum(asset_room, sec_room)
        eligible = room > 1e-12
        if not np.any(eligible):
            break

        mass = float(base_weights[eligible].sum())
        if mass <= 1e-12:
            add = np.zeros_like(out)
            add[eligible] = deficit / int(np.sum(eligible))
        else:
            add = np.zeros_like(out)
            add[eligible] = deficit * (base_weights[eligible] / mass)

        out = out + np.minimum(add, room)

    # Final cleanup: upper bounds + sector caps (numerical tolerance)
    out = np.minimum(out, ub)
    for _ in range(4):
        sector_sums: dict[str, float] = {}
        for i, s in enumerate(secs):
            k = str(s)
            sector_sums[k] = sector_sums.get(k, 0.0) + float(out[i])
        violating = {k: v for k, v in sector_sums.items() if v > sector_cap + 1e-12}
        if not violating:
            break
        for k, total in violating.items():
            idx = np.where(secs.astype(str) == k)[0]
            out[idx] *= sector_cap / total

    return np.maximum(out, 0.0)


def apply_constraints_strict_order(
    risky_weights: np.ndarray,
    nav: float,
    adtv: np.ndarray,
    sectors: list[str] | np.ndarray,
    *,
    risk_off: bool = False,
    max_single: float = 0.10,
    max_sector: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Apply constraints in strict order.

    Order:
    1) liquidity/capacity: position_value <= min(NAV*10%, ADTV*0.05*3)
    2) max single-name
    3) max sector
    4) min cash (10%, risk_off: 20%)
    """
    base = _normalize_positive(risky_weights)
    sec = np.asarray(sectors, dtype=object)
    adtv_arr = np.asarray(adtv, dtype=float)

    # (1) liquidity/capacity cap per name, then renormalize remaining risky names.
    liq_cap = np.minimum(0.10, np.maximum(0.0, adtv_arr * 0.05 * 3.0 / max(nav, 1e-12)))
    w1 = _redistribute_with_upper_bounds(base, liq_cap, target_sum=1.0)

    # (2) max single-name cap while respecting step (1).
    name_cap = np.minimum(liq_cap, max_single)
    w2 = _redistribute_with_upper_bounds(w1, name_cap, target_sum=1.0)

    # (3) max sector cap while preserving step (1)+(2).
    w3 = _apply_sector_caps(w2, sec, sector_cap=max_sector, upper_bounds=name_cap, target_sum=1.0)

    # (4) min cash.
    min_cash = 0.20 if risk_off else 0.10
    invest_cap = 1.0 - min_cash
    invested = float(w3.sum())
    if invested > invest_cap + 1e-12:
        w3 = w3 * (invest_cap / invested)
        invested = float(w3.sum())

    cash_w = max(0.0, 1.0 - invested)
    return w3, cash_w


def apply_no_trade_band(
    target_w: np.ndarray,
    current_w: np.ndarray,
    prices: np.ndarray,
    nav: float,
    *,
    band: float = 0.0025,
    board_lot: int = 100,
    min_notional: float = 5_000_000.0,
) -> np.ndarray:
    t = np.asarray(target_w, dtype=float)
    c = np.asarray(current_w, dtype=float)
    p = np.asarray(prices, dtype=float)
    _validate_same_length(len(t), current_w=c, prices=p)
    delta_w = t - c

    desired_qty = np.floor((np.abs(delta_w) * nav) / np.maximum(p, 1e-12)).astype(int)
    lot_qty = np.array([clamp_qty_to_board_lot(int(q), board_lot=board_lot) for q in desired_qty])
    notionals = lot_qty * p

    keep = (np.abs(delta_w) >= band) & (lot_qty >= board_lot) & (notionals >= min_notional)
    out = c.copy()
    out[keep] = t[keep]
    return out


def cap_turnover(target_w: np.ndarray, current_w: np.ndarray, max_turnover: float = 0.30) -> np.ndarray:
    t = np.asarray(target_w, dtype=float)
    c = np.asarray(current_w, dtype=float)
    delta = t - c
    turnover = float(np.abs(delta).sum() / 2.0)
    if turnover <= max_turnover + 1e-12:
        return t
    scale = max_turnover / turnover
    return c + delta * scale


def generate_trade_intents(
    symbols: list[str],
    target_w: np.ndarray,
    current_w: np.ndarray,
    nav: float,
    next_open_prices: np.ndarray,
    *,
    reasons: dict[str, str] | None = None,
    market_rules_path: str = "configs/market_rules_vn.yaml",
) -> list[dict[str, Any]]:
    t = np.asarray(target_w, dtype=float)
    c = np.asarray(current_w, dtype=float)
    p = np.asarray(next_open_prices, dtype=float)
    if len(symbols) != len(t):
        raise ValueError("symbols length must match target weights")
    _validate_same_length(len(t), current_w=c, next_open_prices=p)
    intents: list[dict[str, Any]] = []

    for i, sym in enumerate(symbols):
        dw = float(t[i] - c[i])
        if abs(dw) <= 0:
            continue
        side = "BUY" if dw > 0 else "SELL"
        raw_qty = int(np.floor(abs(dw) * nav / max(float(p[i]), 1e-12)))
        qty = clamp_qty_to_board_lot(raw_qty, board_lot=100)
        if qty < 100:
            continue
        ref_price = round_price(float(p[i]), side=side, path=market_rules_path)
        intents.append(
            {
                "side": side,
                "symbol": sym,
                "qty": int(qty),
                "ref_price": float(ref_price),
                "target_weight": float(t[i]),
                "reason": (reasons or {}).get(sym, "rebalance_to_target"),
            }
        )
    return intents




def construct_portfolio_v3(
    symbols: list[str],
    returns_252: np.ndarray,
    current_w: np.ndarray,
    nav: float,
    next_open_prices: np.ndarray,
    adtv: np.ndarray,
    atr14: np.ndarray,
    close: np.ndarray,
    spread_proxy: np.ndarray,
    sectors: list[str] | np.ndarray,
    *,
    risk_off: bool = False,
    max_single: float = 0.10,
    max_sector: float = 0.25,
    band: float = 0.0025,
    max_turnover: float = 0.30,
    min_notional: float = 5_000_000.0,
    market_rules_path: str = "configs/market_rules_vn.yaml",
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """End-to-end portfolio construction v3.

    Pipeline:
    1) HRP target (long-only, sum=1)
    2) cost penalty and renormalize
    3) strict constraints order (liq -> single -> sector -> cash)
    4) turnover cap and no-trade rules
    5) trade-intent generation with lot and tick rounding
    """
    if np.asarray(returns_252).ndim != 2:
        raise ValueError("returns_252 must be a 2D matrix with shape (T, N)")
    n_assets = np.asarray(returns_252).shape[1]
    if len(symbols) != n_assets:
        raise ValueError("symbols length must match number of columns in returns_252")

    arr_current = np.asarray(current_w, dtype=float)
    arr_open = np.asarray(next_open_prices, dtype=float)
    arr_adtv = np.asarray(adtv, dtype=float)
    arr_atr = np.asarray(atr14, dtype=float)
    arr_close = np.asarray(close, dtype=float)
    arr_spread = np.asarray(spread_proxy, dtype=float)
    _validate_same_length(
        n_assets,
        current_w=arr_current,
        next_open_prices=arr_open,
        adtv=arr_adtv,
        atr14=arr_atr,
        close=arr_close,
        spread_proxy=arr_spread,
        sectors=np.asarray(sectors, dtype=object),
    )

    hrp_w = compute_hrp_weights(returns_252)
    target_notional = np.asarray(hrp_w, dtype=float) * float(nav)
    penalized_w = apply_cost_penalty_to_weights(
        hrp_w,
        target_notional=target_notional,
        adtv=arr_adtv,
        atr14=arr_atr,
        close=arr_close,
        spread_proxy=arr_spread,
    )
    constrained_w, _cash_w = apply_constraints_strict_order(
        penalized_w,
        nav,
        arr_adtv,
        np.asarray(sectors, dtype=object),
        risk_off=risk_off,
        max_single=max_single,
        max_sector=max_sector,
    )
    final_w, intents = rebalance_with_turnover_and_bands(
        symbols,
        constrained_w,
        current_w,
        nav,
        next_open_prices,
        band=band,
        max_turnover=max_turnover,
        min_notional=min_notional,
        market_rules_path=market_rules_path,
    )
    # Cash must reflect post-trade/no-trade final risky target, not pre-trade constrained target.
    final_cash_w = max(0.0, 1.0 - float(np.sum(final_w)))
    return final_w, final_cash_w, intents

def rebalance_with_turnover_and_bands(
    symbols: list[str],
    target_w: np.ndarray,
    current_w: np.ndarray,
    nav: float,
    next_open_prices: np.ndarray,
    *,
    band: float = 0.0025,
    max_turnover: float = 0.30,
    min_notional: float = 5_000_000.0,
    market_rules_path: str = "configs/market_rules_vn.yaml",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    capped_target = cap_turnover(target_w, current_w, max_turnover=max_turnover)
    filtered_target = apply_no_trade_band(
        capped_target,
        current_w,
        next_open_prices,
        nav,
        band=band,
        min_notional=min_notional,
    )
    intents = generate_trade_intents(
        symbols,
        filtered_target,
        current_w,
        nav,
        next_open_prices,
        market_rules_path=market_rules_path,
    )
    return filtered_target, intents
