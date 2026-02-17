from __future__ import annotations

import pandas as pd


def compute_y_excess(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Compute y, y_vnindex, y_excess using only t and t+horizon."""
    req = {"symbol", "timestamp", "close"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out["as_of_date"] = pd.to_datetime(out["timestamp"]).dt.date
    out["y"] = out.groupby("symbol")["close"].shift(-horizon) / out["close"] - 1.0

    vn = (
        out[out["symbol"] == "VNINDEX"][["timestamp", "close"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .copy()
    )
    if vn.empty:
        cs = out.groupby("timestamp", as_index=False)["close"].mean().sort_values("timestamp")
        vn = cs
    vn["y_vnindex"] = vn["close"].shift(-horizon) / vn["close"] - 1.0
    out = out.merge(vn[["timestamp", "y_vnindex"]], on="timestamp", how="left")
    out["y_excess"] = out["y"] - out["y_vnindex"]
    return out


def compute_rank_z_label(df: pd.DataFrame, col: str = "y_excess") -> pd.DataFrame:
    """Per-date percentile rank then z-score cross-section."""
    out = df.copy()
    out["as_of_date"] = pd.to_datetime(out["timestamp"]).dt.date

    def _rank_z(g: pd.DataFrame) -> pd.Series:
        vals = g[col]
        out_s = pd.Series([float("nan")] * len(g), index=g.index)
        mask = vals.notna()
        if mask.sum() == 0:
            return out_s
        r = vals[mask].rank(pct=True, method="average")
        s = r.std(ddof=0)
        if pd.isna(s) or s == 0:
            out_s.loc[mask] = 0.0
            return out_s
        out_s.loc[mask] = (r - r.mean()) / s
        return out_s

    out["y_rank_z"] = out.groupby("as_of_date", group_keys=False).apply(_rank_z)
    return out
