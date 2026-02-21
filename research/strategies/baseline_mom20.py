from __future__ import annotations

import pandas as pd


def generate_weights(frame: pd.DataFrame, universe: list[str], top_k: int = 3) -> pd.DataFrame:
    f = frame.copy()
    f = f.sort_values(["symbol", "date"])
    f["mom20"] = f.groupby("symbol")["close"].pct_change(20)
    f["week"] = pd.to_datetime(f["date"]).dt.isocalendar().week.astype(int)
    out_rows = []
    for date, g in f.groupby("date"):
        gg = g.sort_values("mom20", ascending=False)
        pick = set(gg.head(top_k)["symbol"].tolist())
        w = 1.0 / max(1, len(pick))
        for _, r in g.iterrows():
            out_rows.append(
                {"date": date, "symbol": r["symbol"], "weight": w if r["symbol"] in pick else 0.0}
            )
    return pd.DataFrame(out_rows)
