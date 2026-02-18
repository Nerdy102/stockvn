from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))

from contracts.canonical import canonical_json, payload_hash
from contracts.schemas import MarketEventV1


def test_canonical_json_key_order_stable() -> None:
    payload_a = {"b": 2, "a": 1, "nested": {"z": 9, "x": 7}}
    payload_b = {"nested": {"x": 7, "z": 9}, "a": 1, "b": 2}

    assert canonical_json(payload_a) == canonical_json(payload_b)
    assert payload_hash(payload_a) == payload_hash(payload_b)


def test_market_event_hash_stable() -> None:
    event = MarketEventV1(
        schema_version="v1",
        event_id="evt-1",
        source="fixture",
        channel="trade",
        ts=dt.datetime(2025, 1, 1, 9, 0, 0),
        payload={"symbol": "AAA", "price": 10.5, "qty": 100},
    )

    expected = "dd9e26559336af99b94981a298e3b9f00f5f2e6356496f107e2158e501ba2770"
    assert event.canonical_payload_hash() == expected


def test_schema_version_required() -> None:
    with pytest.raises(ValueError, match="schema_version is required"):
        MarketEventV1(
            schema_version="",
            event_id="evt-1",
            source="fixture",
            channel="trade",
            ts=dt.datetime(2025, 1, 1, 9, 0, 0),
            payload={"symbol": "AAA"},
        )
