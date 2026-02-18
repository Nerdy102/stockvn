from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))
sys.path.insert(0, str(ROOT / "services" / "market_gateway"))

from market_gateway.normalize import normalize_payload


def test_gateway_event_id_deterministic() -> None:
    raw_a = {
        "event_type": "trade",
        "provider_ts": "2025-01-02T09:00:05Z",
        "symbol": "AAA",
        "exchange": "HOSE",
        "instrument": "EQUITY",
        "price": 10.5,
        "qty": 100,
        "payload": {"k": 1, "x": 2},
    }
    raw_b = {
        "symbol": "AAA",
        "price": 10.5,
        "qty": 100,
        "exchange": "HOSE",
        "instrument": "EQUITY",
        "event_type": "trade",
        "provider_ts": "2025-01-02T09:00:05Z",
        "payload": {"x": 2, "k": 1},
    }
    ev_a = normalize_payload(raw_a, source="demo", channel="replay")
    ev_b = normalize_payload(raw_b, source="demo", channel="replay")
    assert ev_a.event_id == ev_b.event_id
    assert ev_a.payload_hash == ev_b.payload_hash
