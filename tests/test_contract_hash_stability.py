from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))

from contracts.canonical import canonical_json, hash_payload
from contracts.schemas import MarketEventV1


def test_canonical_json_key_order_stable() -> None:
    payload_a = {"b": 2, "a": 1, "nested": {"z": 9, "x": 7}}
    payload_b = {"nested": {"x": 7, "z": 9}, "a": 1, "b": 2}

    assert canonical_json(payload_a) == canonical_json(payload_b)
    assert hash_payload(payload_a) == hash_payload(payload_b)


def test_market_event_hash_deterministic() -> None:
    event = MarketEventV1(
        schema_version="v1",
        source="fixture",
        channel="trade",
        ts=dt.datetime(2025, 1, 1, 9, 0, 0),
        payload={"symbol": "AAA", "price": 10.5, "qty": 100},
    )
    event_same = MarketEventV1(
        schema_version="v1",
        source="fixture",
        channel="trade",
        ts=dt.datetime(2025, 1, 1, 9, 0, 0),
        payload={"qty": 100, "price": 10.5, "symbol": "AAA"},
    )

    assert event.event_id == event_same.event_id
    assert event.payload_hash == event_same.payload_hash


def test_schema_version_required() -> None:
    with pytest.raises(ValueError, match="schema_version is required"):
        MarketEventV1(
            schema_version="",
            source="fixture",
            channel="trade",
            ts=dt.datetime(2025, 1, 1, 9, 0, 0),
            payload={"symbol": "AAA"},
        )
