from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .canonical import payload_hash


@dataclass(frozen=True)
class ContractBase:
    schema_version: str

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version is required")


@dataclass(frozen=True)
class MarketEventV1(ContractBase):
    event_id: str
    source: str
    channel: str
    ts: datetime
    payload: dict[str, Any]

    def canonical_payload_hash(self) -> str:
        body = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "source": self.source,
            "channel": self.channel,
            "ts": self.ts.isoformat(),
            "payload": self.payload,
        }
        return payload_hash(body)


@dataclass(frozen=True)
class CanonicalBar(ContractBase):
    symbol: str
    tf: str
    bar_start: datetime
    bar_end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class TickerSnapshot(ContractBase):
    symbol: str
    ts: datetime
    last_price: float
    total_volume: float


@dataclass(frozen=True)
class QuoteSnapshot(ContractBase):
    symbol: str
    ts: datetime
    best_bid: float | None
    best_ask: float | None
    bid_qty: float | None
    ask_qty: float | None


@dataclass(frozen=True)
class TradePrint(ContractBase):
    symbol: str
    ts: datetime
    price: float
    qty: float
    exchange: str
