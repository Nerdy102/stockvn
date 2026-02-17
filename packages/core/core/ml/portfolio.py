from __future__ import annotations

import numpy as np
import pandas as pd


def apply_constraints(
    sel: pd.DataFrame,
    nav: float,
    max_single: float = 0.10,
    max_sector: float = 0.25,
    min_cash: float = 0.10,
    participation_limit: float = 0.05,
    days_to_exit: int = 3,
) -> pd.DataFrame:
    d = sel.copy()
    d["w"] = d["w_raw"]
    # 1) liquidity per name
    liq_cap = (d["adv20_value"] * participation_limit * days_to_exit) / nav
    d["w"] = np.minimum(d["w"], liq_cap.fillna(0.0))
    # 2) max single
    d["w"] = np.minimum(d["w"], max_single)
    # 3) max sector
    if "sector" not in d:
        d["sector"] = "OTHER"
    sec_sum = d.groupby("sector")["w"].transform("sum").replace(0, np.nan)
    scale = np.minimum(1.0, max_sector / sec_sum).fillna(1.0)
    d["w"] = d["w"] * scale
    # 4) min cash buffer
    invested = d["w"].sum()
    cap = max(0.0, 1.0 - min_cash)
    if invested > cap and invested > 0:
        d["w"] = d["w"] * (cap / invested)
    return d


def form_portfolio(pred: pd.DataFrame, topk: int = 30, nav: float = 1.0) -> pd.DataFrame:
    d = pred.copy()
    d["y_z"] = d.groupby("as_of_date")["y_hat"].transform(lambda s: (s - s.mean()) / (s.std() or 1.0))
    d = d.sort_values(["as_of_date", "y_z"], ascending=[True, False]).groupby("as_of_date").head(topk)
    d["vol_60d"] = d["vol_60d"].replace(0, np.nan)
    d["w_raw"] = (1.0 / d["vol_60d"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["w_raw"] = d.groupby("as_of_date")["w_raw"].transform(lambda s: s / max(1e-12, s.sum()))
    out = []
    for dte, grp in d.groupby("as_of_date"):
        out.append(apply_constraints(grp, nav=nav))
    return pd.concat(out, ignore_index=True) if out else d.iloc[0:0]
