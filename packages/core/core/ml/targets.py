from __future__ import annotations

import pandas as pd

from core.calendar_vn import get_trading_calendar_vn


HORIZON = 21


def compute_y_excess(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Compute y, y_vnindex, y_excess from point-in-time close data."""
    required = {"symbol", "timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    cal = get_trading_calendar_vn()
    out = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["as_of_date"] = out["timestamp"].dt.date
    out["realized_date"] = out["as_of_date"].map(lambda d: cal.shift_trading_days(d, horizon))

    future_close = out[["symbol", "as_of_date", "close"]].rename(
        columns={"as_of_date": "realized_date", "close": "close_t_plus_h"}
    )
    out = out.merge(future_close, on=["symbol", "realized_date"], how="left")
    out["y"] = out["close_t_plus_h"] / out["close"] - 1.0

    vn = out[out["symbol"] == "VNINDEX"][["as_of_date", "close"]].drop_duplicates("as_of_date")
    if vn.empty:
        vn = out.groupby("as_of_date", as_index=False)["close"].mean()
    vn = vn.sort_values("as_of_date")
    vn["realized_date"] = vn["as_of_date"].map(lambda d: cal.shift_trading_days(d, horizon))
    vn_future = vn[["as_of_date", "close"]].rename(columns={"as_of_date": "realized_date", "close": "close_t_plus_h"})
    vn = vn.merge(vn_future, on="realized_date", how="left")
    vn["y_vnindex"] = vn["close_t_plus_h"] / vn["close"] - 1.0
    out = out.merge(vn[["as_of_date", "y_vnindex"]], on="as_of_date", how="left")
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
