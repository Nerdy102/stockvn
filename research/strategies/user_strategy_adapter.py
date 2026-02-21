from __future__ import annotations

import pandas as pd


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    # black-box substitute: use existing features-compatible momentum + volatility penalty
    f = frame.copy().sort_values(["symbol", "date"])
    f["mom10"] = f.groupby("symbol")["close"].pct_change(10)
    f["vol10"] = (
        f.groupby("symbol")["close"].pct_change().rolling(10).std().reset_index(level=0, drop=True)
    )
    f["score"] = f["mom10"].fillna(0.0) - 0.5 * f["vol10"].fillna(0.0)
    rows = []
    for date, g in f.groupby("date"):
        gg = g.sort_values("score", ascending=False).head(3)
        pick = set(gg["symbol"].tolist())
        w = 1.0 / max(1, len(pick))
        for _, r in g.iterrows():
            rows.append(
                {"date": date, "symbol": r["symbol"], "weight": w if r["symbol"] in pick else 0.0}
            )
    return pd.DataFrame(rows)
