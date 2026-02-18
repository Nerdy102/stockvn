from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pandas as pd

from contracts.canonical import derive_event_id


def _parse_payload_col(bronze_df: pd.DataFrame) -> pd.DataFrame:
    if bronze_df.empty:
        return bronze_df
    out = bronze_df.copy()
    out["payload_obj"] = out["payload_json"].apply(
        lambda x: json.loads(str(x)) if isinstance(x, str) else x
    )
    return out


def _to_utc_ts(v: Any) -> str:
    ts = pd.Timestamp(v)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def bronze_to_canonical_trades(bronze_df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_payload_col(bronze_df)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        p = row["payload_obj"]
        symbol = str(p.get("symbol", ""))
        ts_utc = _to_utc_ts(p.get("ts_utc") or p.get("ts") or row["received_ts_utc"])
        price = float(p.get("price", 0.0))
        qty = float(p.get("qty", 0.0))
        if price <= 0 or qty <= 0 or not symbol:
            continue
        event_id = derive_event_id(
            {
                "source": row["source"],
                "channel": row["channel"],
                "symbol": symbol,
                "ts_utc": ts_utc,
                "price": price,
                "qty": qty,
                "payload_hash": row["payload_hash"],
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "source": str(row["source"]),
                "symbol": symbol,
                "exchange": str(p.get("exchange", "UNKNOWN")),
                "instrument": str(p.get("instrument", "EQUITY")),
                "ts_utc": ts_utc,
                "price": price,
                "qty": qty,
                "payload_hash": str(row["payload_hash"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.drop_duplicates(subset=["event_id"])
        .sort_values(["symbol", "ts_utc", "event_id"])
        .reset_index(drop=True)
    )


def bronze_to_canonical_quotes(bronze_df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_payload_col(bronze_df)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        p = row["payload_obj"]
        if "bid_px" not in p and "ask_px" not in p:
            continue
        symbol = str(p.get("symbol", ""))
        ts_utc = _to_utc_ts(p.get("ts_utc") or p.get("ts") or row["received_ts_utc"])
        event_id = derive_event_id(
            {
                "source": row["source"],
                "symbol": symbol,
                "ts_utc": ts_utc,
                "payload_hash": row["payload_hash"],
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "source": str(row["source"]),
                "symbol": symbol,
                "ts_utc": ts_utc,
                "bid_px": float(p.get("bid_px", 0.0)),
                "bid_qty": float(p.get("bid_qty", 0.0)),
                "ask_px": float(p.get("ask_px", 0.0)),
                "ask_qty": float(p.get("ask_qty", 0.0)),
                "payload_hash": str(row["payload_hash"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.drop_duplicates(subset=["event_id"])
        .sort_values(["symbol", "ts_utc", "event_id"])
        .reset_index(drop=True)
    )


def compute_silver_dq_metrics(silver_trades: pd.DataFrame) -> dict[str, float]:
    if silver_trades.empty:
        return {"missing_rate": 0.0, "duplicate_rate": 0.0, "ohlc_invariant_rate": 0.0, "psi": 0.0}
    total = float(len(silver_trades))
    missing = (
        silver_trades[["symbol", "ts_utc", "price", "qty", "payload_hash"]].isna().any(axis=1).sum()
    )
    dup = total - float(len(silver_trades.drop_duplicates(subset=["event_id"])))
    return {
        "missing_rate": float(missing / total),
        "duplicate_rate": float(dup / total),
        "ohlc_invariant_rate": 0.0,
        "psi": 0.0,
    }


def dq_incident_severity(metrics: dict[str, float]) -> str | None:
    if metrics.get("missing_rate", 0.0) > 0.05:
        return "HIGH"
    if metrics.get("duplicate_rate", 0.0) > 0.005:
        return "MEDIUM"
    return None
