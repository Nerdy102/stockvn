from __future__ import annotations

import pandas as pd


def _smoothed_scores(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy().sort_values(["symbol", "date"])
    f["ret1"] = f.groupby("symbol")["close"].pct_change()
    f["mom10"] = f.groupby("symbol")["close"].pct_change(10)
    f["vol10"] = f.groupby("symbol")["ret1"].rolling(10).std().reset_index(level=0, drop=True)
    f["score_raw"] = f["mom10"].fillna(0.0) - 0.5 * f["vol10"].fillna(0.0)
    f["score_smooth"] = (
        f.groupby("symbol")["score_raw"]
        .ewm(alpha=0.2, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return f


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    top_k = 3
    buffer = 2
    no_trade_band = 0.02
    f = _smoothed_scores(frame)
    dates = sorted(f["date"].unique())
    prev = {s: 0.0 for s in universe}
    rows: list[dict] = []

    for i, d in enumerate(dates):
        day = f[f["date"] == d].copy()
        target = dict(prev)
        if i % 5 == 0:
            day = day.sort_values("score_smooth", ascending=False).reset_index(drop=True)
            day["rank"] = range(1, len(day) + 1)
            current_hold = {s for s, w in prev.items() if w > 0.0}
            selected = []
            for r in day.itertuples():
                keep = r.rank <= top_k
                if str(r.symbol) in current_hold:
                    keep = keep or (r.rank <= (top_k + buffer) and float(r.score_smooth) > 0.0)
                if keep:
                    selected.append(str(r.symbol))
            selected = selected[: max(top_k + buffer, len(selected))]
            w = 1.0 / max(1, len(selected))
            target = {s: (w if s in set(selected) else 0.0) for s in universe}
            for s in universe:
                if abs(target.get(s, 0.0) - prev.get(s, 0.0)) < no_trade_band:
                    target[s] = prev.get(s, 0.0)
            norm = sum(max(0.0, target[s]) for s in universe)
            if norm > 0:
                for s in universe:
                    target[s] = max(0.0, target[s]) / norm

        for s in universe:
            rows.append({"date": d, "symbol": s, "weight": float(target.get(s, 0.0))})
        prev = target

    return pd.DataFrame(rows)
