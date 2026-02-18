from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))

from lakehouse.silver import bronze_to_canonical_trades


def test_silver_validation_rejects_non_positive_price_qty() -> None:
    bronze = pd.DataFrame(
        [
            {
                "source": "demo",
                "channel": "trade",
                "received_ts_utc": "2025-01-02T09:00:00Z",
                "payload_json": '{"symbol":"AAA","ts_utc":"2025-01-02T09:00:00Z","price":0,"qty":10}',
                "payload_hash": "h1",
            },
            {
                "source": "demo",
                "channel": "trade",
                "received_ts_utc": "2025-01-02T09:00:01Z",
                "payload_json": '{"symbol":"AAA","ts_utc":"2025-01-02T09:00:01Z","price":10,"qty":-1}',
                "payload_hash": "h2",
            },
        ]
    )

    out = bronze_to_canonical_trades(bronze)
    assert out.empty
