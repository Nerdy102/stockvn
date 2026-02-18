from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .canonical import derive_event_id, hash_payload, strict_mapping


def _require_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _require_dt(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    return value


@dataclass(frozen=True)
class ContractBase:
    schema_version: str
    payload_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty("schema_version", self.schema_version)


@dataclass(frozen=True)
class MarketEventV1(ContractBase):
    source: str
    channel: str
    ts: datetime
    payload: Mapping[str, Any]
    event_id: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_non_empty("source", self.source)
        _require_non_empty("channel", self.channel)
        _require_dt("ts", self.ts)
        normalized_payload = strict_mapping(self.payload, field_name="payload")
        body = {
            "schema_version": self.schema_version,
            "source": self.source,
            "channel": self.channel,
            "ts": self.ts.isoformat(),
            "payload": normalized_payload,
        }
        event_id = self.event_id or derive_event_id(body)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "payload_hash", hash_payload({**body, "event_id": event_id}))


@dataclass(frozen=True)
class CanonicalBarV1(ContractBase):
    symbol: str
    timeframe: str
    bar_start: datetime
    bar_end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    event_id: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_non_empty("symbol", self.symbol)
        _require_non_empty("timeframe", self.timeframe)
        _require_dt("bar_start", self.bar_start)
        _require_dt("bar_end", self.bar_end)
        body = {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_start": self.bar_start.isoformat(),
            "bar_end": self.bar_end.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }
        event_id = self.event_id or derive_event_id(body)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "payload_hash", hash_payload({**body, "event_id": event_id}))


@dataclass(frozen=True)
class ProviderSnapshot(ContractBase):
    provider: str
    snapshot_type: str
    ts: datetime
    payload: Mapping[str, Any]
    event_id: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_non_empty("provider", self.provider)
        _require_non_empty("snapshot_type", self.snapshot_type)
        _require_dt("ts", self.ts)
        normalized_payload = strict_mapping(self.payload, field_name="payload")
        body = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "snapshot_type": self.snapshot_type,
            "ts": self.ts.isoformat(),
            "payload": normalized_payload,
        }
        event_id = self.event_id or derive_event_id(body)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "payload_hash", hash_payload({**body, "event_id": event_id}))


# Backward-compatible aliases.
CanonicalBar = CanonicalBarV1
