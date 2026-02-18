from __future__ import annotations

from typing import Any

from .base import MarketProviderAdapter


class SsiStubAdapter(MarketProviderAdapter):
    """Provider-agnostic stub for SSI-like payloads.

    Expected raw fields (documented only):
    - event_type, provider_ts, symbol, exchange, instrument, price, qty, payload
    This adapter yields no events unless fixture_events are injected.
    """

    def __init__(self, fixture_events: list[dict[str, Any]] | None = None) -> None:
        self.fixture_events = fixture_events or []

    def iter_raw_events(self) -> list[dict[str, Any]]:
        return list(self.fixture_events)
