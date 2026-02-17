from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_flags(base: pd.DataFrame, regime: pd.DataFrame | None) -> pd.DataFrame:
    out = base.copy()
    out["regime_trend_up"] = 0.0
    out["regime_sideways"] = 1.0
    out["regime_risk_off"] = 0.0
    if regime is None or regime.empty:
        return out

    rg = regime.copy()
    rg["as_of_date"] = pd.to_datetime(rg["as_of_date"]).dt.date
    out = out.merge(rg[["as_of_date", "regime"]], on="as_of_date", how="left")
    out["regime_trend_up"] = (out["regime"] == "trend_up").astype(float)
    out["regime_sideways"] = out["regime"].fillna("sideways").eq("sideways").astype(float)
    out["regime_risk_off"] = (out["regime"] == "risk_off").astype(float)
    return out.drop(columns=["regime"])


def compute_foreign_flow_features(foreign_daily: pd.DataFrame, adv_reference: pd.DataFrame) -> pd.DataFrame:
    if foreign_daily.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util"])
    f = foreign_daily.copy().sort_values(["symbol", "as_of_date"])
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).dt.date
    f["net_foreign_val"] = f["net_foreign_val"].astype(float)
    f["net_foreign_val_5d"] = f.groupby("symbol")["net_foreign_val"].rolling(5).sum().reset_index(level=0, drop=True)
    f["net_foreign_val_20d"] = f.groupby("symbol")["net_foreign_val"].rolling(20).sum().reset_index(level=0, drop=True)

    out = f.merge(adv_reference[["symbol", "as_of_date", "adv20_value"]], on=["symbol", "as_of_date"], how="left")
    out["foreign_flow_intensity"] = out["net_foreign_val_20d"] / out["adv20_value"].replace(0, np.nan)
    out["foreign_room_util"] = 1.0 - out.get("current_room", np.nan) / out.get("total_room", np.nan)
    return out[["symbol", "as_of_date", "net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util"]]


def _quote_level_values(row: pd.Series, side: str, i: int) -> tuple[float, float]:
    price_col = f"{side}Price{i}"
    vol_col = f"{side}Vol{i}"
    if price_col in row and vol_col in row:
        return float(row.get(price_col, 0.0) or 0.0), float(row.get(vol_col, 0.0) or 0.0)

    book = row.get("bids" if side == "bid" else "asks")
    if isinstance(book, dict):
        prices = book.get("prices", [])
        vols = book.get("volumes", [])
        if len(prices) >= i and len(vols) >= i:
            return float(prices[i - 1]), float(vols[i - 1])
    return 0.0, 0.0


def compute_orderbook_daily_features(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "imb_1_day", "imb_3_day", "spread_day"])
    q = quotes.copy()
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    q["as_of_date"] = q["timestamp"].dt.date

    rows = []
    for _, r in q.iterrows():
        bp1, bv1 = _quote_level_values(r, "bid", 1)
        ap1, av1 = _quote_level_values(r, "ask", 1)
        bid3 = ask3 = 0.0
        for i in (1, 2, 3):
            _, bv = _quote_level_values(r, "bid", i)
            _, av = _quote_level_values(r, "ask", i)
            bid3 += bv
            ask3 += av
        mid = (ap1 + bp1) / 2.0 if (ap1 + bp1) > 0 else np.nan
        rows.append(
            {
                "symbol": r.get("symbol"),
                "as_of_date": r["as_of_date"],
                "imb_1": (bv1 - av1) / (bv1 + av1 + 1e-9),
                "imb_3": (bid3 - ask3) / (bid3 + ask3 + 1e-9),
                "spread": (ap1 - bp1) / mid if pd.notna(mid) else np.nan,
            }
        )
    tmp = pd.DataFrame(rows)
    return tmp.groupby(["symbol", "as_of_date"], as_index=False).agg(
        imb_1_day=("imb_1", "mean"),
        imb_3_day=("imb_3", "mean"),
        spread_day=("spread", "mean"),
    )


def compute_intraday_daily_features(intraday_1m: pd.DataFrame) -> pd.DataFrame:
    if intraday_1m.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "rv_day", "vol_first_hour_ratio"])
    b = intraday_1m.copy().sort_values(["symbol", "timestamp"])
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    b["as_of_date"] = b["timestamp"].dt.date
    b["time"] = b["timestamp"].dt.time
    b["r_1m"] = b.groupby(["symbol", "as_of_date"])["close"].pct_change().fillna(0.0)

    daily = b.groupby(["symbol", "as_of_date"], as_index=False).agg(
        rv_day=("r_1m", lambda x: float(np.sqrt((x**2).sum()))),
        vol_total_day=("volume", "sum"),
    )
    first_hour = b[(b["time"] >= pd.to_datetime("09:15").time()) & (b["time"] <= pd.to_datetime("10:15").time())]
    fh = first_hour.groupby(["symbol", "as_of_date"], as_index=False).agg(vol_first_hour=("volume", "sum"))
    out = daily.merge(fh, on=["symbol", "as_of_date"], how="left")
    out["vol_first_hour_ratio"] = out["vol_first_hour"].fillna(0.0) / out["vol_total_day"].replace(0, np.nan)
    return out[["symbol", "as_of_date", "rv_day", "vol_first_hour_ratio"]]


def build_features_v2(
    base_features: pd.DataFrame,
    regime_df: pd.DataFrame | None,
    foreign_df: pd.DataFrame | None,
    quotes_df: pd.DataFrame | None,
    intraday_df: pd.DataFrame | None,
) -> pd.DataFrame:
    out = base_features.copy()
    out = add_regime_flags(out, regime_df)

    if foreign_df is not None and not foreign_df.empty:
        ff = compute_foreign_flow_features(foreign_df, out[["symbol", "as_of_date", "adv20_value"]].drop_duplicates())
        out = out.merge(ff, on=["symbol", "as_of_date"], how="left")
    else:
        for c in ["net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util"]:
            out[c] = np.nan

    ob = compute_orderbook_daily_features(quotes_df if quotes_df is not None else pd.DataFrame())
    out = out.merge(ob, on=["symbol", "as_of_date"], how="left")
    iv = compute_intraday_daily_features(intraday_df if intraday_df is not None else pd.DataFrame())
    out = out.merge(iv, on=["symbol", "as_of_date"], how="left")

    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        out[c] = out[c].fillna(out.groupby("as_of_date")[c].transform("median"))
    return out
