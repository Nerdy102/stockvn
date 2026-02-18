from __future__ import annotations

import json
from typing import Any

import pandas as pd


def lineage_payload_hashes(df: pd.DataFrame) -> list[str]:
    if "payload_hash" not in df.columns or df.empty:
        return []
    return sorted({str(x) for x in df["payload_hash"].dropna().astype(str).tolist()})


def attach_lineage(record: dict[str, Any], *, trades_df: pd.DataFrame) -> dict[str, Any]:
    out = dict(record)
    out["lineage_payload_hashes_json"] = json.dumps(
        lineage_payload_hashes(trades_df), separators=(",", ":")
    )
    out["lineage_event_ids_json"] = json.dumps(
        sorted(
            {
                str(x)
                for x in trades_df.get("event_id", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .tolist()
            }
        ),
        separators=(",", ":"),
    )
    return out
