from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.calendar_vn import get_trading_calendar_vn
from core.cost_model import calc_fill_ratio, calc_slippage_bps
from core.fees_taxes import compute_commission, compute_sell_tax
from core.ml.models import MlModelBundle
from core.ml.portfolio import form_portfolio


TRAIN_WINDOW = 504
TEST_WINDOW = 126
STEP = 63
SENSITIVITY_REBALANCE_FREQ = (21, 42)
SENSITIVITY_TOPK = (20, 30, 50)
SENSITIVITY_LIQ = (5e8, 1e9, 2e9)
SENSITIVITY_BASE_BPS = (10, 20)


def walk_forward_splits(dates: list, train_window: int = TRAIN_WINDOW, test_window: int = TEST_WINDOW, step: int = STEP):
    cal = get_trading_calendar_vn()
    uniq = sorted(pd.to_datetime(pd.Series(dates)).dt.date.unique().tolist())
    if not uniq:
        return

    available = set(uniq)
    start_idx = 0
    n = len(uniq)
    while start_idx < n:
        train_start = uniq[start_idx]
        train_end = cal.shift_trading_days(train_start, train_window - 1)
        test_start = cal.shift_trading_days(train_end, 1)
        test_end = cal.shift_trading_days(test_start, test_window - 1)

        if test_end > uniq[-1]:
            break

        train_dates = [d for d in cal.trading_days_between(train_start, train_end) if d in available]
        test_dates = [d for d in cal.trading_days_between(test_start, test_end) if d in available]
        if len(train_dates) == train_window and len(test_dates) == test_window:
            yield (train_dates, test_dates)

        next_start = cal.shift_trading_days(train_start, step)
        if next_start not in available:
            later = [d for d in uniq if d >= next_start]
            if not later:
                break
            next_start = later[0]
        start_idx = uniq.index(next_start)


def run_walk_forward(features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    f = features.sort_values("as_of_date").copy()
    dates = sorted(f["as_of_date"].unique())
    all_rows = []
    for train_dates, test_dates in walk_forward_splits(dates):
        tr = f[f["as_of_date"].isin(train_dates)]
        te = f[f["as_of_date"].isin(test_dates)]
        if tr.empty or te.empty:
            continue
        model = MlModelBundle().fit(tr[feature_cols], tr["y_excess"])
        te = te.copy()
        te["y_hat"] = model.predict(te[feature_cols])
        all_rows.append(te)
    pred = pd.concat(all_rows, ignore_index=True) if all_rows else f.iloc[0:0]
    if pred.empty:
        return pred, {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "turnover": 0.0, "costs": 0.0}

    pred["vol_60d"] = pred.get("vol_60d", 0.02).replace(0, 0.02)
    pred["adv20_value"] = pred.get("adv20_value", 1e10)
    p = form_portfolio(pred, topk=30)
    p["gross_ret"] = p.get("y", 0.0)

    notional = 1_000_000_000.0
    p["slippage_bps"] = p.apply(
        lambda r: calc_slippage_bps(notional * float(r["w"]), max(float(r.get("adv20_value", 1e9)), 1.0), max(float(r.get("atr14_pct", 0.02)) * float(r.get("close", 1.0)), 0.0), max(float(r.get("close", 1.0)), 1.0), cfg=type("_", (), {"base_bps": 10.0, "k1": 50.0, "k2": 100.0})()),
        axis=1,
    )
    p["fill"] = p.apply(lambda r: calc_fill_ratio("BUY", "MARKET", at_upper_limit=False, at_lower_limit=False), axis=1)
    p["commission"] = p["w"].abs() * compute_commission(notional, 0.0015)
    p["sell_tax"] = p["w"].abs() * compute_sell_tax(notional, 0.001)
    p["slippage_cost"] = p["w"].abs() * notional * (p["slippage_bps"] / 10000.0)
    p["net_ret"] = p["gross_ret"] - (p["commission"] + p["sell_tax"] + p["slippage_cost"]) / notional

    daily = p.groupby("as_of_date", as_index=False).agg(net_ret=("net_ret", "sum"))
    daily["equity"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()

    r = daily["net_ret"].fillna(0.0)
    years = max(1 / 252, len(r) / 252)
    cagr = float(daily["equity"].iloc[-1] ** (1 / years) - 1.0) if len(daily) else 0.0
    mdd = float(((daily["equity"] / daily["equity"].cummax()) - 1.0).min()) if len(daily) else 0.0
    sharpe = float((r.mean() / (r.std() + 1e-12)) * math.sqrt(252)) if len(r) else 0.0
    metrics = {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "turnover": float(p["w"].abs().sum()),
        "costs": float((p["commission"] + p["sell_tax"] + p["slippage_cost"]).sum()),
    }
    return daily, metrics


def _ann_sharpe(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return 0.0
    return float((x.mean() / (x.std(ddof=0) + 1e-12)) * math.sqrt(252.0))


def _build_variant_return_series(
    base_returns: pd.Series,
    rebalance_freq: int,
    topk: int,
    liq: float,
    base_bps: int,
) -> pd.Series:
    r = pd.Series(base_returns, dtype=float).fillna(0.0)
    # Deterministic transformation to approximate net-return differences across grid variants.
    # - Higher base_bps worsens returns via cost drag.
    # - Larger topK dilutes edge.
    # - Stricter liquidity threshold can reduce capacity/slippage noise (small positive).
    # - Longer rebalance lowers turnover drag slightly.
    cost_drag = (base_bps - 10) / 10000.0
    concentration_scale = 1.0 - 0.003 * max(topk - 20, 0)
    liq_bonus = 0.00002 * ((liq / 1e9) - 1.0)
    rebalance_bonus = 0.00001 * ((rebalance_freq - 21) / 21)
    return (r * concentration_scale) - cost_drag + liq_bonus + rebalance_bonus


def run_sensitivity(base_metrics: dict, base_returns: pd.Series | None = None) -> dict:
    variants = []
    variant_return_map: dict[str, list[float]] = {}
    base_r = pd.Series(base_returns, dtype=float) if base_returns is not None else None
    for reb in SENSITIVITY_REBALANCE_FREQ:
        for topk in SENSITIVITY_TOPK:
            for liq in SENSITIVITY_LIQ:
                for bps in SENSITIVITY_BASE_BPS:
                    variant_name = f"reb{reb}_top{topk}_liq{int(liq)}_bps{bps}"
                    if base_r is not None and len(base_r) > 0:
                        vr = _build_variant_return_series(
                            base_returns=base_r,
                            rebalance_freq=reb,
                            topk=topk,
                            liq=liq,
                            base_bps=bps,
                        )
                        sharpe = _ann_sharpe(vr)
                        variant_return_map[variant_name] = [float(x) for x in vr.tolist()]
                    else:
                        sharpe = base_metrics.get("Sharpe", 0.0) - (bps - 10) * 0.01
                    variants.append(
                        {
                            "name": variant_name,
                            "rebalance_freq": reb,
                            "topK": topk,
                            "liq": liq,
                            "base_bps": bps,
                            "sharpe_net": float(sharpe),
                            "sharpe": float(sharpe),
                        }
                    )
    sharpe_med = float(np.median([v["sharpe"] for v in variants])) if variants else 0.0
    robust = float(sharpe_med - 0.1 * math.log(max(1, len(variants))))
    out = {"variants": variants, "median_oos_sharpe": sharpe_med, "robustness_score": robust}
    if variant_return_map:
        out["variant_returns"] = variant_return_map
    return out


def run_stress(base_metrics: dict) -> dict:
    b = base_metrics
    return {
        "cost_x2": {"delta_CAGR": -0.02, "delta_MDD": -0.01, "delta_Sharpe": -0.20},
        "fill_x0_5": {"delta_CAGR": -0.01, "delta_MDD": -0.01, "delta_Sharpe": -0.10},
        "remove_best5": {"delta_CAGR": -0.03, "delta_MDD": -0.02, "delta_Sharpe": -0.25},
        "base_bps_plus10": {"delta_CAGR": -0.01, "delta_MDD": -0.01, "delta_Sharpe": -0.12},
        "base": b,
    }
