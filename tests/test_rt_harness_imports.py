from __future__ import annotations

import hashlib
import json

from tools.realtime_harness.generate_synthetic_events import generate_synthetic_events


def _hash_events(events: list[dict[str, object]]) -> str:
    payload = json.dumps(events, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_generate_module_import_and_deterministic_output() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    first = generate_synthetic_events(symbols, days=2, seed=42)
    second = generate_synthetic_events(symbols, days=2, seed=42)
    assert _hash_events(first) == _hash_events(second)
    assert first[0]["event_id"]
    assert first[0]["payload_hash"]
