from __future__ import annotations

import numpy as np
import pandas as pd


def build_weights_ivp_uncertainty(selected: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    out = selected.copy()
    out["vol_60d"] = out["vol_60d"].replace(0, np.nan).fillna(1.0)
    out["uncert"] = out["uncert"].fillna(out["uncert"].median() if "uncert" in out else 0.0)
    out["w"] = 1.0 / (out["vol_60d"] + eps)
    med_u = float(np.nanmedian(out["uncert"])) if len(out) else 1.0
    med_u = med_u if med_u > 0 else 1.0
    out["w"] = out["w"] / (1.0 + out["uncert"] / med_u)
    s = float(out["w"].sum())
    out["w"] = out["w"] / s if s > 0 else 0.0
    return out


def apply_constraints_ordered(
    selected: pd.DataFrame,
    nav: float,
    risk_off: bool = False,
    max_single: float = 0.10,
    max_sector: float = 0.25,
) -> pd.DataFrame:
    out = selected.copy()
    min_cash = 0.20 if risk_off else 0.10

    # (1) liquidity capacity and NAV single cap
    liq_cap = (out["adv20_value"].fillna(0.0) * 0.05 * 3.0) / max(nav, 1.0)
    out["w"] = np.minimum(out["w"], liq_cap)
    out["liq_bound"] = out["w"] <= liq_cap
    out["w"] = np.minimum(out["w"], max_single)

    # (2) max single
    out["w"] = np.minimum(out["w"], max_single)

    # (3) max sector
    out["sector"] = out.get("sector", "OTHER").fillna("OTHER")
    sec_sum = out.groupby("sector")["w"].transform("sum").replace(0, np.nan)
    sec_scale = np.minimum(1.0, max_sector / sec_sum).fillna(1.0)
    out["w"] = out["w"] * sec_scale

    # (4) min cash
    invest_cap = 1.0 - min_cash
    invested = float(out["w"].sum())
    if invested > invest_cap and invested > 0:
        out["w"] *= invest_cap / invested

    out["cash_w"] = max(0.0, 1.0 - float(out["w"].sum()))
    return out


def apply_exposure_overlay(selected: pd.DataFrame, regime: str) -> pd.DataFrame:
    mult = 1.0
    if regime == "sideways":
        mult = 0.8
    elif regime == "risk_off":
        mult = 0.5
    out = selected.copy()
    out["w"] = out["w"] * mult
    out["cash_w"] = max(0.0, 1.0 - float(out["w"].sum()))
    return out


def apply_no_trade_band(orders: pd.DataFrame) -> pd.DataFrame:
    out = orders.copy()
    out["current_w"] = out.get("current_w", 0.0).fillna(0.0)
    out["target_w"] = out.get("target_w", out.get("w", 0.0)).fillna(0.0)
    out["qty"] = out.get("qty", out.get("target_qty", 0)).fillna(0)
    out["order_notional"] = out.get("order_notional", 0.0).fillna(0.0)

    keep = (
        (out["target_w"] - out["current_w"]).abs() >= 0.0025
    ) & (out["qty"] >= 100) & (out["order_notional"] >= 5_000_000)
    return out[keep].copy()
