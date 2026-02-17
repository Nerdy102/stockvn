from __future__ import annotations

import pandas as pd


HORIZON = 21


def compute_y_excess(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Compute y, y_vnindex, y_excess from point-in-time close data."""
    required = {"symbol", "timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["as_of_date"] = out["timestamp"].dt.date
    out["y"] = out.groupby("symbol")["close"].shift(-horizon) / out["close"] - 1.0

    vn = out[out["symbol"] == "VNINDEX"][["timestamp", "close"]].drop_duplicates("timestamp")
    if vn.empty:
        vn = out.groupby("timestamp", as_index=False)["close"].mean()
    vn = vn.sort_values("timestamp")
    vn["y_vnindex"] = vn["close"].shift(-horizon) / vn["close"] - 1.0
    out = out.merge(vn[["timestamp", "y_vnindex"]], on="timestamp", how="left")
    out["y_excess"] = out["y"] - out["y_vnindex"]
    return out


def compute_rank_z_label(df: pd.DataFrame, col: str = "y_excess") -> pd.DataFrame:
    """Per-date percentile rank followed by cross-sectional z-score."""
    out = df.copy()
    if "timestamp" in out.columns:
        out["as_of_date"] = pd.to_datetime(out["timestamp"]).dt.date

    labels = pd.Series(index=out.index, dtype=float)
    for _, idx in out.groupby("as_of_date").groups.items():
        vals = out.loc[idx, col]
        valid = vals.notna()
        if valid.sum() == 0:
            labels.loc[idx] = float("nan")
            continue
        ranks = vals[valid].rank(pct=True, method="average")
        sigma = float(ranks.std(ddof=0))
        z = (ranks - ranks.mean()) / sigma if sigma > 0 else 0.0
        labels.loc[idx] = float("nan")
        labels.loc[ranks.index] = z

    out["y_rank_z"] = labels
    return out
