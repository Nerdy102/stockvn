from __future__ import annotations

import numpy as np
import pandas as pd


def build_weights_ivp_uncertainty(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("score_final", ascending=False).head(30).copy()
    out["inv_vol"] = 1.0 / (out["vol_60d"].fillna(0.0) + eps)
    m_unc = float(out["uncert"].median()) if len(out) else 1.0
    if m_unc <= 0:
        m_unc = 1.0
    out["w"] = out["inv_vol"] / (1.0 + out["uncert"].fillna(m_unc) / m_unc)
    s = float(out["w"].sum())
    out["w"] = out["w"] / s if s > 0 else 0.0
    return out


def apply_constraints_ordered(df: pd.DataFrame, nav: float, risk_off: bool = False) -> pd.DataFrame:
    out = df.copy()
    min_cash = 0.2 if risk_off else 0.1
    out["w_cap_liq"] = np.minimum(
        out["adv20_value"].fillna(0.0) * 0.05 * 3.0 / max(nav, 1.0),
        0.10,
    )
    out["w"] = np.minimum(out["w"], out["w_cap_liq"])
    out["w"] = np.minimum(out["w"], 0.10)

    if "sector" in out.columns:
        for sec, idx in out.groupby("sector").groups.items():
            tot = float(out.loc[idx, "w"].sum())
            if tot > 0.25:
                out.loc[idx, "w"] *= 0.25 / tot

    investable = 1.0 - min_cash
    wsum = float(out["w"].sum())
    if wsum > investable and wsum > 0:
        out["w"] *= investable / wsum
    out["cash_w"] = max(0.0, 1.0 - float(out["w"].sum()))
    return out


def apply_no_trade_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["current_w"] = out.get("current_w", 0.0)
    out["delta_w"] = (out["w"] - out["current_w"]).abs()
    out["target_qty"] = out.get("target_qty", 0)
    out["order_notional"] = out.get("order_notional", 0.0)
    mask = (out["delta_w"] >= 0.0025) & (out["target_qty"] >= 100) & (out["order_notional"] >= 5_000_000)
    return out[mask].copy()
