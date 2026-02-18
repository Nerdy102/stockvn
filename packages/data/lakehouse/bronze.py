from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.canonical import hash_payload


def _normalize_envelope(record: dict[str, Any]) -> dict[str, Any]:
    required = ["source", "channel", "received_ts_utc", "schema_version", "payload_json"]
    for key in required:
        if key not in record:
            raise ValueError(f"missing required bronze key: {key}")

    payload = record["payload_json"]
    if not isinstance(payload, dict):
        raise TypeError("payload_json must be object")

    payload_h = str(record.get("payload_hash") or hash_payload(payload))
    received = pd.Timestamp(record["received_ts_utc"], tz="UTC")
    trace_json = record.get("trace_json") if isinstance(record.get("trace_json"), dict) else {}
    return {
        "source": str(record["source"]),
        "channel": str(record["channel"]),
        "received_ts_utc": received.isoformat(),
        "schema_version": str(record["schema_version"]),
        "payload_json": json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "payload_hash": payload_h,
        "trace_json": json.dumps(
            trace_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "dt": received.date().isoformat(),
    }


def append_bronze_records(base_dir: str | Path, records: list[dict[str, Any]]) -> Path | None:
    if not records:
        return None
    rows = [_normalize_envelope(r) for r in records]
    df = pd.DataFrame(rows)
    first = rows[0]
    part_dir = (
        Path(base_dir)
        / f"dt={first['dt']}"
        / f"source={first['source']}"
        / f"channel={first['channel']}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    out = part_dir / "records.parquet"

    if out.exists():
        old = pd.read_parquet(out)
        merged = pd.concat([old, df], ignore_index=True)
    else:
        merged = df

    merged = merged.drop_duplicates(subset=["payload_hash"]).sort_values("received_ts_utc")
    merged.to_parquet(out, index=False)
    return out


def read_bronze_partition(
    base_dir: str | Path,
    *,
    dt_value: dt.date,
    source: str,
    channel: str,
) -> pd.DataFrame:
    path = (
        Path(base_dir)
        / f"dt={dt_value.isoformat()}"
        / f"source={source}"
        / f"channel={channel}"
        / "records.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def verify_bronze_hashes(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    for _, row in df.iterrows():
        payload = json.loads(str(row["payload_json"]))
        if hash_payload(payload) != str(row["payload_hash"]):
            return False
    return True
