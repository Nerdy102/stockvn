from __future__ import annotations

from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events
from tools.realtime_harness.verify_invariants import verify_invariants


def test_invariant_checker_catches_duplicate() -> None:
    events = [e.__dict__ for e in generate_synthetic_events(symbols=10, days=1, seed=42)]
    events.append(dict(events[0]))
    out = verify_invariants(events)
    # duplicate raw events should be visible in checker counters
    assert out["duplicate_event_count"] > 0
