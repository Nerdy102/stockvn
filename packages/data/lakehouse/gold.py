from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from contracts.canonical import hash_payload
from .lineage import lineage_payload_hashes


def build_bars_from_trades(trades_df: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    if timeframe not in {"15m", "60m"}:
        raise ValueError("timeframe must be 15m or 60m")

    freq = "15min" if timeframe == "15m" else "60min"
    df = trades_df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    out_rows: list[dict[str, object]] = []
    for symbol, sdf in df.groupby("symbol"):
        sdf = sdf.sort_values("ts_utc")
        for bucket, bdf in sdf.groupby(pd.Grouper(key="ts_utc", freq=freq)):
            if bdf.empty:
                continue
            start_ts = pd.Timestamp(bucket)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            end_ts = start_ts + (
                pd.Timedelta(minutes=15) if timeframe == "15m" else pd.Timedelta(hours=1)
            )
            row = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_ts": start_ts.isoformat(),
                "end_ts": end_ts.isoformat(),
                "o": float(bdf.iloc[0]["price"]),
                "h": float(bdf["price"].max()),
                "l": float(bdf["price"].min()),
                "c": float(bdf.iloc[-1]["price"]),
                "v": float(bdf["qty"].sum()),
                "n_trades": int(len(bdf)),
                "vwap": float((bdf["price"] * bdf["qty"]).sum() / max(1e-12, bdf["qty"].sum())),
                "finalized": True,
                "build_ts": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "lineage_payload_hashes_json": json.dumps(
                    lineage_payload_hashes(bdf), separators=(",", ":")
                ),
            }
            row["bar_hash"] = hash_payload(row)
            out_rows.append(row)
    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    return out.sort_values(["symbol", "start_ts"]).reset_index(drop=True)


def build_feature_snapshots(bars_df: pd.DataFrame) -> pd.DataFrame:
    if bars_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in bars_df.iterrows():
        features = {
            "ret_1": 0.0,
            "vwap_to_close": (
                (float(row["vwap"]) / float(row["c"])) if float(row["c"]) != 0 else 0.0
            ),
            "n_trades": int(row["n_trades"]),
        }
        snap = {
            "as_of_ts": row["end_ts"],
            "symbol": row["symbol"],
            "timeframe": row["timeframe"],
            "features_json": json.dumps(features, sort_keys=True, separators=(",", ":")),
            "lineage_json": json.dumps(
                {
                    "lineage_payload_hashes_json": row["lineage_payload_hashes_json"],
                    "bar_hash": row["bar_hash"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "as_of_date": str(pd.Timestamp(row["end_ts"]).date()),
            "matured_date": str(pd.Timestamp(row["end_ts"]).date()),
            "public_date": str(pd.Timestamp(row["end_ts"]).date()),
        }
        snap["feature_hash"] = hash_payload(snap)
        rows.append(snap)
    return pd.DataFrame(rows).sort_values(["symbol", "as_of_ts"]).reset_index(drop=True)
