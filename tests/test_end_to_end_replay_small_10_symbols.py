from __future__ import annotations

from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events
from tools.realtime_harness.replay_to_redis import InMemoryRedisStream
from tools.realtime_harness.verify_invariants import verify_invariants


def test_end_to_end_replay_small_10_symbols() -> None:
    events = [e.__dict__ for e in generate_synthetic_events(symbols=10, days=1, seed=42)]
    redis = InMemoryRedisStream()
    for ev in events:
        redis.xadd("stream:market_events", {"payload": str(ev)})
    assert len(redis.xrange("stream:market_events")) == len(events)

    out = verify_invariants(events)
    assert out["bars_hash_deterministic"] is True
    assert out["signals_idempotent"] is True
