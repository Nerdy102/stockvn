from __future__ import annotations

import pandas as pd


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    f = frame.copy().sort_values(["symbol", "date"])
    f["sma20"] = f.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())
    f["sma50"] = f.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    f["signal"] = (f["sma20"] > f["sma50"]).astype(float)
    out = f[["date", "symbol", "signal"]].rename(columns={"signal": "weight"})
    sums = out.groupby("date")["weight"].transform("sum").replace(0.0, 1.0)
    out["weight"] = out["weight"] / sums
    return out
