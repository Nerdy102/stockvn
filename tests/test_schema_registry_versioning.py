from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))

from contracts.models import MarketEventV1
from contracts.registry import SchemaRegistry, build_default_registry


def test_default_registry_contains_contracts() -> None:
    registry = build_default_registry()

    assert registry.resolve("MarketEvent", "v1") is MarketEventV1
    assert registry.versions("MarketEvent") == ["v1"]


def test_registry_disallows_conflicting_reregistration() -> None:
    registry = SchemaRegistry()
    registry.register("MarketEvent", "v1", MarketEventV1)

    class FakeSchema:
        schema_version = "v1"
        ts = dt.datetime(2025, 1, 1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("MarketEvent", "v1", FakeSchema)


def test_registry_missing_schema_raises() -> None:
    registry = SchemaRegistry()
    with pytest.raises(KeyError, match="schema not found"):
        registry.resolve("Unknown", "v1")
