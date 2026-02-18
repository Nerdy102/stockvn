from __future__ import annotations

import json
from pathlib import Path

from tools.realtime_harness.generator import generate_events
from tools.realtime_harness.invariants import check_invariants


def test_generator_deterministic() -> None:
    a = generate_events(seed=42, symbols=5, days=1)
    b = generate_events(seed=42, symbols=5, days=1)
    assert a == b


def test_invariant_checker_catches_duplicate() -> None:
    events = generate_events(seed=42, symbols=2, days=1)
    events.append(events[0])
    result = check_invariants(events)
    assert result.ok is False
    assert result.duplicate_count > 0


def test_replayable_evidence_pack_files_exist_and_parse() -> None:
    base = Path("tests/fixtures/replay")
    event_lines = (
        (base / "event_log_fixture.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(event_lines) > 0
    json.loads((base / "expected_bars_fixture.json").read_text(encoding="utf-8"))
    json.loads((base / "expected_signals_fixture.json").read_text(encoding="utf-8"))
    parity = json.loads(
        (base / "expected_parity_reconciliation_fixture.json").read_text(encoding="utf-8")
    )
    assert parity["equity_parity_ok"] is True
