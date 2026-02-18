from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "market_gateway"))

from market_gateway.adapters.replay import ReplayAdapter
from market_gateway.dedup_store import DedupStore
from market_gateway.event_log import EventLogWriter
from market_gateway.main import MarketGateway
from market_gateway.publish import StreamPublisher
from tests.helpers_redis_fake import FakeRedisCompat


def test_replay_publishes_ordered(tmp_path) -> None:
    redis = FakeRedisCompat()
    adapter = ReplayAdapter("tests/fixtures/event_log_tiny.jsonl", replay_sorted=True)
    gw = MarketGateway(
        mode="replay",
        adapter=adapter,
        publisher=StreamPublisher(redis, stream_prefix="market"),
        dedup_store=DedupStore(str(tmp_path / "dedup.sqlite3")),
        event_log=EventLogWriter(str(tmp_path / "elog"), rotate_every=10),
        source="demo",
        channel="replay",
    )
    out = gw.run_once()
    assert out["published"] == 2

    rows = redis.xrange("market:trade")
    provider_ts = []
    for _, fields in rows:
        payload = json.loads(fields["payload"])
        provider_ts.append(payload["provider_ts"])
    assert provider_ts == sorted(provider_ts)
