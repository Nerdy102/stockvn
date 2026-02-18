from __future__ import annotations

import datetime as dt

from core.db.models import EventLog
from scripts.replay_events import replay_into_redis
from tests.helpers_redis_fake import FakeRedisCompat


def test_replay_preserves_order_and_counts() -> None:
    events = [
        EventLog(id=1, ts_utc=dt.datetime(2025, 1, 6, 9, 0, 0), source="t", event_type="signal", symbol="AAA", payload_json={"target": 1}, payload_hash="h1"),
        EventLog(id=2, ts_utc=dt.datetime(2025, 1, 6, 9, 0, 1), source="t", event_type="bar", symbol="AAA", payload_json={"open": 10.0}, payload_hash="h2"),
        EventLog(id=3, ts_utc=dt.datetime(2025, 1, 6, 9, 0, 2), source="t", event_type="bar", symbol="BBB", payload_json={"open": 20.0}, payload_hash="h3"),
    ]
    redis_client = FakeRedisCompat()

    count = replay_into_redis(events, redis_client, speed="max")

    assert count == 3
    entries = redis_client.xrange("ssi:bar")
    assert len(entries) == 2
    assert entries[0][1]["symbol"] == "AAA"
    assert entries[1][1]["symbol"] == "BBB"
