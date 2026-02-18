from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "market_gateway"))

from market_gateway.adapters.ssi_stub import SsiStubAdapter
from market_gateway.dedup_store import DedupStore
from market_gateway.event_log import EventLogWriter
from market_gateway.main import MarketGateway
from market_gateway.publish import StreamPublisher
from tests.helpers_redis_fake import FakeRedisCompat


def test_gateway_dedup_skips_duplicates(tmp_path) -> None:
    fixture = [
        {
            "event_type": "trade",
            "provider_ts": "2025-01-02T09:00:05Z",
            "symbol": "AAA",
            "exchange": "HOSE",
            "instrument": "EQUITY",
            "price": 10.5,
            "qty": 100,
        },
        {
            "event_type": "trade",
            "provider_ts": "2025-01-02T09:00:05Z",
            "symbol": "AAA",
            "exchange": "HOSE",
            "instrument": "EQUITY",
            "price": 10.5,
            "qty": 100,
        },
    ]
    redis = FakeRedisCompat()
    gw = MarketGateway(
        mode="provider",
        adapter=SsiStubAdapter(fixture),
        publisher=StreamPublisher(redis, stream_prefix="market"),
        dedup_store=DedupStore(str(tmp_path / "dedup.sqlite3")),
        event_log=EventLogWriter(str(tmp_path / "elog"), rotate_every=10),
        source="demo",
        channel="trade",
    )
    out = gw.run_once()
    assert out["consumed"] == 2
    assert out["published"] == 1
    assert out["duplicates"] == 1
