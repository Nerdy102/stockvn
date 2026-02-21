from __future__ import annotations

import pandas as pd


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    out = frame[["date", "symbol"]].copy()
    d = pd.to_datetime(out["date"])
    month_start = d.dt.to_period("M").dt.to_timestamp()
    ew = 1.0 / max(1, len(universe))
    out["weight"] = ew
    out["rebalance_key"] = month_start
    return out.drop(columns=["rebalance_key"])
