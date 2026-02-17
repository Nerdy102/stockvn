from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_flags(features: pd.DataFrame, regime_df: pd.DataFrame | None = None) -> pd.DataFrame:
    out = features.copy()
    out["regime_trend_up"] = 0.0
    out["regime_sideways"] = 1.0
    out["regime_risk_off"] = 0.0
    if regime_df is not None and not regime_df.empty:
        tmp = regime_df.copy()
        tmp["as_of_date"] = pd.to_datetime(tmp["as_of_date"]).dt.date
        out = out.merge(tmp[["as_of_date", "regime"]], on="as_of_date", how="left")
        out["regime_trend_up"] = (out["regime"] == "trend_up").astype(float)
        out["regime_sideways"] = (out["regime"].fillna("sideways") == "sideways").astype(float)
        out["regime_risk_off"] = (out["regime"] == "risk_off").astype(float)
        out = out.drop(columns=["regime"])
    return out


def compute_foreign_flow_features(meta_df: pd.DataFrame, adv_df: pd.DataFrame) -> pd.DataFrame:
    df = meta_df.copy().sort_values(["symbol", "as_of_date"])
    g = df.groupby("symbol", group_keys=False)
    df["net_foreign_val_5d"] = g["net_foreign_val"].rolling(5).sum().reset_index(level=0, drop=True)
    df["net_foreign_val_20d"] = g["net_foreign_val"].rolling(20).sum().reset_index(level=0, drop=True)
    out = df.merge(adv_df[["symbol", "as_of_date", "adv20_value"]], on=["symbol", "as_of_date"], how="left")
    out["foreign_flow_intensity"] = out["net_foreign_val_20d"] / out["adv20_value"].replace(0, np.nan)
    out["foreign_room_util"] = 1.0 - out["current_room"] / out["total_room"].replace(0, np.nan)
    return out[["symbol", "as_of_date", "net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util"]]


def compute_orderbook_daily_features(quotes_df: pd.DataFrame) -> pd.DataFrame:
    if quotes_df.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "imb_1_day", "imb_3_day", "spread_day"])
    q = quotes_df.copy()
    q["as_of_date"] = pd.to_datetime(q["timestamp"]).dt.date
    q["imb_1"] = (q["bidVol1"] - q["askVol1"]) / (q["bidVol1"] + q["askVol1"] + 1e-9)
    q["imb_3"] = (q[["bidVol1", "bidVol2", "bidVol3"]].sum(axis=1) - q[["askVol1", "askVol2", "askVol3"]].sum(axis=1)) / (
        q[["bidVol1", "bidVol2", "bidVol3"]].sum(axis=1) + q[["askVol1", "askVol2", "askVol3"]].sum(axis=1) + 1e-9
    )
    mid = (q["askPrice1"] + q["bidPrice1"]) / 2.0
    q["spread"] = (q["askPrice1"] - q["bidPrice1"]) / mid.replace(0, np.nan)
    g = q.groupby(["symbol", "as_of_date"], as_index=False).agg(
        imb_1_day=("imb_1", "mean"),
        imb_3_day=("imb_3", "mean"),
        spread_day=("spread", "mean"),
    )
    return g


def compute_intraday_daily_features(bars_1m: pd.DataFrame) -> pd.DataFrame:
    if bars_1m.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "rv_day", "vol_first_hour_ratio"])
    b = bars_1m.copy().sort_values(["symbol", "timestamp"])
    b["as_of_date"] = pd.to_datetime(b["timestamp"]).dt.date
    b["time"] = pd.to_datetime(b["timestamp"]).dt.time
    b["r_1m"] = b.groupby(["symbol", "as_of_date"])["close"].pct_change().fillna(0.0)

    day = b.groupby(["symbol", "as_of_date"], as_index=False).agg(
        rv_day=("r_1m", lambda x: float(np.sqrt(np.square(x).sum()))),
        vol_total_day=("volume", "sum"),
    )
    fh = b[(b["time"] >= pd.to_datetime("09:15").time()) & (b["time"] <= pd.to_datetime("10:15").time())]
    fh = fh.groupby(["symbol", "as_of_date"], as_index=False).agg(vol_first_hour=("volume", "sum"))
    out = day.merge(fh, on=["symbol", "as_of_date"], how="left")
    out["vol_first_hour_ratio"] = out["vol_first_hour"].fillna(0.0) / out["vol_total_day"].replace(0, np.nan)
    return out[["symbol", "as_of_date", "rv_day", "vol_first_hour_ratio"]]
