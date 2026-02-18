from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
from scipy.optimize import linprog

from core.market_rules import clamp_qty_to_board_lot, round_price

from core.alpha_v3.costs import apply_cost_penalty_to_weights
from core.alpha_v3.hrp import compute_hrp_weights



def _validate_same_length(n: int, **arrays: np.ndarray) -> None:
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"{name} length {len(arr)} does not match expected {n}")


def _normalize_positive(weights: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) * (float(target_sum) / max(1, len(w)))
    return (w / s) * float(target_sum)


def _redistribute_with_upper_bounds(
    base_weights: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    target_sum: float,
) -> np.ndarray:
    w = _normalize_positive(base_weights, target_sum=target_sum)
    ub = np.maximum(np.asarray(upper_bounds, dtype=float), 0.0)
    feasible_total = min(float(target_sum), float(ub.sum()))
    if feasible_total <= 0:
        return np.zeros_like(w)

    out = np.minimum(w, ub)

    for _ in range(10 * len(w) + 20):
        deficit = feasible_total - float(out.sum())
        if deficit <= 1e-12:
            break
        room = ub - out
        eligible = room > 1e-12
        if not np.any(eligible):
            break

        mass = float(w[eligible].sum())
        add = np.zeros_like(out)
        if mass <= 1e-12:
            add[eligible] = deficit / int(np.sum(eligible))
        else:
            add[eligible] = deficit * (w[eligible] / mass)
        out = np.minimum(out + add, ub)

    return np.maximum(out, 0.0)


def _project_sum(x: np.ndarray, target_sum: float) -> np.ndarray:
    return np.asarray(x, dtype=float) + (float(target_sum) - float(np.sum(x))) / max(1, len(x))


def _project_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def _project_sector_halfspace(x: np.ndarray, idx: np.ndarray, cap: float) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    if len(idx) == 0:
        return out
    sec_sum = float(np.sum(out[idx]))
    if sec_sum <= cap:
        return out
    shift = (sec_sum - cap) / float(len(idx))
    out[idx] -= shift
    return out


def dykstra_project_weights(
    base_weights: np.ndarray,
    *,
    target_sum: float,
    upper_bounds: np.ndarray,
    sectors: list[str] | np.ndarray,
    sector_cap: float,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    base = np.asarray(base_weights, dtype=float)
    ub = np.maximum(np.asarray(upper_bounds, dtype=float), 0.0)
    sec = np.asarray(sectors, dtype=object)
    if len(base) == 0:
        return np.array([], dtype=float)

    x = _normalize_positive(base, target_sum=min(float(target_sum), float(ub.sum())))
    x = np.minimum(np.maximum(x, 0.0), ub)

    sector_keys = sorted({str(s) for s in sec})
    sector_indices = [np.where(sec.astype(str) == k)[0] for k in sector_keys]

    n_constraints = 2 + len(sector_indices)
    increments = [np.zeros_like(x) for _ in range(n_constraints)]

    lower = np.zeros_like(x)
    for _ in range(max_iter):
        x_prev = x.copy()

        y = x + increments[0]
        x = _project_box(y, lower, ub)
        increments[0] = y - x

        y = x + increments[1]
        x = _project_sum(y, target_sum=float(target_sum))
        increments[1] = y - x

        for j, idx in enumerate(sector_indices, start=2):
            y = x + increments[j]
            x = _project_sector_halfspace(y, idx, cap=float(sector_cap))
            increments[j] = y - x

        if np.linalg.norm(x - x_prev, ord=2) <= tol:
            break

    # deterministic cleanup to satisfy constraints numerically
    for _ in range(20):
        x = _project_box(x, lower, ub)
        for idx in sector_indices:
            x = _project_sector_halfspace(x, idx, cap=float(sector_cap))
        x = _project_sum(x, target_sum=float(target_sum))
        if (
            float(np.max(np.maximum(0.0, -x))) <= 1e-9
            and abs(float(np.sum(x)) - float(target_sum)) <= 1e-9
            and float(np.max(np.maximum(0.0, x - ub))) <= 1e-9
            and max((float(np.sum(x[idx])) for idx in sector_indices), default=0.0) <= float(sector_cap) + 1e-9
        ):
            break

    x = np.minimum(np.maximum(x, 0.0), ub)
    x = _redistribute_with_upper_bounds(x, ub, target_sum=min(float(target_sum), float(ub.sum())))

    # final sector trim + re-redistribute to preserve exact risky sum when feasible
    for idx in sector_indices:
        sec_sum = float(np.sum(x[idx]))
        if sec_sum > float(sector_cap) + 1e-12:
            x[idx] *= float(sector_cap) / sec_sum
    x = _redistribute_with_upper_bounds(x, ub, target_sum=min(float(target_sum), float(ub.sum())))

    return np.maximum(x, 0.0)


def _constraint_violations(
    w: np.ndarray,
    *,
    target_sum: float,
    upper_bounds: np.ndarray,
    sectors: np.ndarray,
    sector_cap: float,
) -> dict[str, float]:
    vec = np.asarray(w, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)
    sec = np.asarray(sectors, dtype=object)
    sec_keys = sorted({str(s) for s in sec})
    sec_excess = 0.0
    for k in sec_keys:
        idx = np.where(sec.astype(str) == k)[0]
        sec_excess = max(sec_excess, max(0.0, float(np.sum(vec[idx])) - float(sector_cap)))

    return {
        "nonneg": float(np.max(np.maximum(0.0, -vec))),
        "sum": float(abs(float(np.sum(vec)) - float(target_sum))),
        "name_cap": float(np.max(np.maximum(0.0, vec - ub))),
        "sector_cap": float(sec_excess),
    }


def build_constraint_report(
    *,
    base_weights: np.ndarray,
    projected_weights: np.ndarray,
    target_sum: float,
    upper_bounds: np.ndarray,
    sectors: list[str] | np.ndarray,
    sector_cap: float,
    cash_target: float,
) -> dict[str, Any]:
    base = np.asarray(base_weights, dtype=float)
    post = np.asarray(projected_weights, dtype=float)
    sec = np.asarray(sectors, dtype=object)

    pre = _constraint_violations(
        base,
        target_sum=target_sum,
        upper_bounds=np.asarray(upper_bounds, dtype=float),
        sectors=sec,
        sector_cap=sector_cap,
    )
    post_v = _constraint_violations(
        post,
        target_sum=target_sum,
        upper_bounds=np.asarray(upper_bounds, dtype=float),
        sectors=sec,
        sector_cap=sector_cap,
    )
    eps = 1e-12
    base_pos = np.maximum(base, 0.0)
    base_pos = _normalize_positive(base_pos, target_sum=max(float(np.sum(post)), eps))
    post_pos = np.maximum(post, 0.0)

    active = [
        "nonneg",
        "sum_risky",
        "name_cap",
        "sector_cap",
        "cash_target",
    ]

    return {
        "active_constraints": active,
        "cash_target": float(cash_target),
        "pre_violations": pre,
        "post_violations": post_v,
        "distance": {
            "l2": float(np.linalg.norm(post - base, ord=2)),
            "kl": float(np.sum(post_pos * np.log((post_pos + eps) / (base_pos + eps)))),
        },
    }


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
    base = _normalize_positive(risky_weights, target_sum=1.0)
    sec = np.asarray(sectors, dtype=object)
    adtv_arr = np.asarray(adtv, dtype=float)

    liq_cap = np.maximum(0.0, adtv_arr * 0.05 * 3.0 / max(nav, 1e-12))
    name_cap = np.minimum(np.minimum(liq_cap, max_single), 1.0)

    cash_target = 0.20 if risk_off else 0.10
    risky_target = max(0.0, 1.0 - cash_target)
    projected = dykstra_project_weights(
        base,
        target_sum=risky_target,
        upper_bounds=name_cap,
        sectors=sec,
        sector_cap=max_sector,
    )

    cash_w = max(0.0, 1.0 - float(np.sum(projected)))
    return projected, cash_w


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


def apply_cvar_overlay(
    returns_252: np.ndarray,
    base_weights: np.ndarray,
    *,
    alpha: float = 0.05,
    lower_mult: float = 0.5,
    upper_mult: float = 1.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    rets = np.asarray(returns_252, dtype=float)
    base = _normalize_positive(base_weights, target_sum=1.0)
    n = len(base)
    if n == 0:
        return base, {"status": "empty"}

    lb = np.maximum(0.0, lower_mult * base)
    ub = np.minimum(1.0, upper_mult * base)
    if float(lb.sum()) > 1.0 + 1e-12 or float(ub.sum()) < 1.0 - 1e-12:
        fallback = _redistribute_with_upper_bounds(base, ub, target_sum=min(1.0, float(ub.sum())))
        return fallback, {"status": "fallback_bounds_infeasible"}

    t = rets.shape[0]
    if t <= 1:
        clipped = np.clip(base, lb, ub)
        return _normalize_positive(clipped, target_sum=1.0), {"status": "fallback_short_history"}

    c = np.zeros(n + 1 + t, dtype=float)
    c[n] = 1.0
    c[n + 1 :] = 1.0 / (max(1e-12, (1.0 - alpha) * t))

    a_eq = np.zeros((1, n + 1 + t), dtype=float)
    a_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    a_ub = np.zeros((t, n + 1 + t), dtype=float)
    b_ub = np.zeros(t, dtype=float)
    for i in range(t):
        # loss_i - z - u_i <= 0, loss_i = -r_i @ w
        a_ub[i, :n] = -rets[i]
        a_ub[i, n] = -1.0
        a_ub[i, n + 1 + i] = -1.0

    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(float(lb[i]), float(ub[i])) for i in range(n)])
    bounds.append((None, None))
    bounds.extend([(0.0, None) for _ in range(t)])

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        clipped = np.clip(base, lb, ub)
        fallback = _normalize_positive(clipped, target_sum=1.0)
        return fallback, {"status": "fallback_lp_failed", "message": str(res.message)}

    w = np.maximum(res.x[:n], 0.0)
    w = _normalize_positive(w, target_sum=1.0)
    w = np.clip(w, lb, ub)
    w = _normalize_positive(w, target_sum=1.0)
    return w, {"status": "optimal", "fun": float(res.fun)}


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


def _select_topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    k = int(min(max(1, top_k), len(s)))
    return np.argsort(-s, kind="mergesort")[:k]


def construct_portfolio_v3_with_report(
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
    scores: np.ndarray | None = None,
    top_k: int = 30,
    risk_off: bool = False,
    max_single: float = 0.10,
    max_sector: float = 0.25,
    band: float = 0.0025,
    max_turnover: float = 0.30,
    min_notional: float = 5_000_000.0,
    market_rules_path: str = "configs/market_rules_vn.yaml",
) -> tuple[np.ndarray, float, list[dict[str, Any]], dict[str, Any]]:
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
    arr_sectors = np.asarray(sectors, dtype=object)
    _validate_same_length(
        n_assets,
        current_w=arr_current,
        next_open_prices=arr_open,
        adtv=arr_adtv,
        atr14=arr_atr,
        close=arr_close,
        spread_proxy=arr_spread,
        sectors=arr_sectors,
    )

    if scores is None:
        idx = np.arange(min(top_k, n_assets))
    else:
        idx = _select_topk_indices(np.asarray(scores, dtype=float), top_k=top_k)

    symbols_k = [symbols[int(i)] for i in idx]
    ret_k = np.asarray(returns_252)[:, idx]
    cur_k = arr_current[idx]
    open_k = arr_open[idx]
    adtv_k = arr_adtv[idx]
    atr_k = arr_atr[idx]
    close_k = arr_close[idx]
    spread_k = arr_spread[idx]
    sectors_k = arr_sectors[idx]

    hrp_w = compute_hrp_weights(ret_k)
    target_notional = np.asarray(hrp_w, dtype=float) * float(nav)
    penalized_w = apply_cost_penalty_to_weights(
        hrp_w,
        target_notional=target_notional,
        adtv=adtv_k,
        atr14=atr_k,
        close=close_k,
        spread_proxy=spread_k,
    )

    cvar_w, cvar_info = apply_cvar_overlay(ret_k, penalized_w, alpha=0.05, lower_mult=0.5, upper_mult=1.5)

    liq_cap = np.maximum(0.0, adtv_k * 0.05 * 3.0 / max(nav, 1e-12))
    name_cap = np.minimum(np.minimum(liq_cap, max_single), 1.0)
    cash_target = 0.20 if risk_off else 0.10
    risky_target = max(0.0, 1.0 - cash_target)

    constrained_w = dykstra_project_weights(
        cvar_w,
        target_sum=risky_target,
        upper_bounds=name_cap,
        sectors=sectors_k,
        sector_cap=max_sector,
    )

    constrained_w = _normalize_positive(constrained_w, target_sum=risky_target)
    final_w, intents = rebalance_with_turnover_and_bands(
        symbols_k,
        constrained_w,
        cur_k,
        nav,
        open_k,
        band=band,
        max_turnover=max_turnover,
        min_notional=min_notional,
        market_rules_path=market_rules_path,
    )

    final_cash_w = max(0.0, 1.0 - float(np.sum(final_w)))
    constraint_report = build_constraint_report(
        base_weights=cvar_w,
        projected_weights=constrained_w,
        target_sum=risky_target,
        upper_bounds=name_cap,
        sectors=sectors_k,
        sector_cap=max_sector,
        cash_target=cash_target,
    )
    constraint_report["cvar_overlay"] = cvar_info
    constraint_report["universe"] = {
        "top_k": int(len(symbols_k)),
        "symbols": symbols_k,
    }

    return final_w, final_cash_w, intents, constraint_report


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
    final_w, final_cash_w, intents, _report = construct_portfolio_v3_with_report(
        symbols,
        returns_252,
        current_w,
        nav,
        next_open_prices,
        adtv,
        atr14,
        close,
        spread_proxy,
        sectors,
        risk_off=risk_off,
        max_single=max_single,
        max_sector=max_sector,
        band=band,
        max_turnover=max_turnover,
        min_notional=min_notional,
        market_rules_path=market_rules_path,
    )
    return final_w, final_cash_w, intents


def persist_constraint_report(
    db: Any,
    *,
    as_of_date: dt.date,
    report: dict[str, Any],
    run_tag: str = "alpha_v3",
) -> None:
    from core.db.models import RebalanceConstraintReport

    db.add(
        RebalanceConstraintReport(
            as_of_date=as_of_date,
            run_tag=str(run_tag),
            report_json=dict(report),
        )
    )
    db.commit()
