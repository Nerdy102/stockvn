from __future__ import annotations

import pandas as pd

from core.raocmoe import load_config


def generate_weights(frame: pd.DataFrame, universe: list[str], variant: str) -> pd.DataFrame:
    _ = load_config()
    f = frame.copy().sort_values(["symbol", "date"])
    f["mom5"] = f.groupby("symbol")["close"].pct_change(5).fillna(0.0)
    penalty = {
        "RAOCMOE_FULL": 0.20,
        "RAOCMOE_minus_D1": 0.15,
        "RAOCMOE_minus_D2": 0.15,
        "RAOCMOE_minus_D3": 0.10,
        "RAOCMOE_minus_D4": 0.05,
        "RAOCMOE_minus_D5": 0.05,
        "RAOCMOE_minus_D6": 0.05,
    }.get(variant, 0.20)
    f["score"] = f["mom5"] - penalty * f.groupby("symbol")["close"].pct_change().abs().fillna(0.0)
    rows = []
    for date, g in f.groupby("date"):
        pick_df = g.sort_values("score", ascending=False).head(3)
        pick = set(pick_df["symbol"].tolist())
        w = 1.0 / max(1, len(pick))
        for _, r in g.iterrows():
            rows.append(
                {"date": date, "symbol": r["symbol"], "weight": w if r["symbol"] in pick else 0.0}
            )
    return pd.DataFrame(rows)
