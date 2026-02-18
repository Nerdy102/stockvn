from __future__ import annotations

import datetime as dt

import pandas as pd

HORIZON = 21
LABEL_VERSION = "v3"


def build_labels_v3(prices: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Build label table v3 at date t from forward horizon H returns."""
    required = {"symbol", "timestamp", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = prices.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    g = df.groupby("symbol", group_keys=False)
    df["ret_stock_21d"] = g["close"].shift(-horizon) / df["close"] - 1.0

    vn = df[df["symbol"] == "VNINDEX"][["timestamp", "close"]].drop_duplicates("timestamp")
    if vn.empty:
        vn = df.groupby("timestamp", as_index=False)["close"].mean()
    vn = vn.sort_values("timestamp")
    vn["ret_vnindex_21d"] = vn["close"].shift(-horizon) / vn["close"] - 1.0

    out = df.merge(vn[["timestamp", "ret_vnindex_21d"]], on="timestamp", how="left")
    out["y_excess"] = out["ret_stock_21d"] - out["ret_vnindex_21d"]

    labels = pd.Series(index=out.index, dtype=float)
    for _, idx in out.groupby("date").groups.items():
        vals = out.loc[idx, "y_excess"]
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
    out["label_version"] = LABEL_VERSION
    return out[["symbol", "date", "y_excess", "y_rank_z", "label_version"]]


def assert_no_label_overlap(
    features_date: dt.date,
    label_date: dt.date,
    horizon: int = HORIZON,
) -> None:
    """
    Hard-block leakage: label timestamp must be at least H days after features_date.

    The canonical non-leaky setup is `label_date == features_date + horizon`.
    """
    min_label_date = features_date + dt.timedelta(days=horizon)
    if label_date < min_label_date:
        raise RuntimeError(
            "Leakage detected: "
            f"label_date={label_date} is earlier than min allowed {min_label_date} "
            f"for features_date={features_date}, horizon={horizon}."
        )
