from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))

from lakehouse.gold import build_bars_from_trades


def test_lineage_payload_hashes_present_in_gold_bars() -> None:
    trades = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "symbol": "AAA",
                "ts_utc": "2025-01-02T09:00:00Z",
                "price": 10.0,
                "qty": 100.0,
                "payload_hash": "h1",
            },
            {
                "event_id": "e2",
                "symbol": "AAA",
                "ts_utc": "2025-01-02T09:05:00Z",
                "price": 11.0,
                "qty": 200.0,
                "payload_hash": "h2",
            },
        ]
    )
    bars = build_bars_from_trades(trades, timeframe="15m")
    assert not bars.empty
    lineage = json.loads(str(bars.iloc[0]["lineage_payload_hashes_json"]))
    assert lineage == ["h1", "h2"]
