from __future__ import annotations

from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events


def test_generator_deterministic() -> None:
    a = generate_synthetic_events(symbols=10, days=2, seed=42)
    b = generate_synthetic_events(symbols=10, days=2, seed=42)
    assert a == b
