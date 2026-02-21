from __future__ import annotations

import pandas as pd


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    out = frame[["date", "symbol"]].copy()
    ew = 1.0 / max(1, len(universe))
    out["weight"] = out["symbol"].map({s: ew for s in universe}).fillna(0.0)
    return out
