from __future__ import annotations

import datetime as dt
import json
from hashlib import sha256
from typing import Any

from contracts.canonical import hash_payload
from contracts.models import MarketEventV1


def _utc(ts: str | dt.datetime) -> dt.datetime:
    p = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if p.tzinfo is None:
        p = p.replace(tzinfo=dt.timezone.utc)
    else:
        p = p.astimezone(dt.timezone.utc)
    return p.replace(tzinfo=None)


def normalize_payload(
    raw: dict[str, Any], *, source: str, channel: str, schema_version: str = "v1"
) -> MarketEventV1:
    provider_ts = raw.get("provider_ts") or raw.get("ts")
    if provider_ts is None:
        raise ValueError("provider_ts/ts is required")
    symbol = str(raw.get("symbol", "")).strip()
    if not symbol:
        raise ValueError("symbol is required")
    payload = {
        "event_type": str(raw.get("event_type", "trade")),
        "provider_ts": _utc(provider_ts).isoformat() + "Z",
        "symbol": symbol,
        "exchange": str(raw.get("exchange", "UNKNOWN")),
        "instrument": str(raw.get("instrument", "EQUITY")),
        "price": float(raw.get("price", 0.0)),
        "qty": float(raw.get("qty", 0.0)),
        "payload": raw.get("payload", raw),
    }
    if payload["price"] <= 0 or payload["qty"] <= 0:
        raise ValueError("price and qty must be > 0")

    event = MarketEventV1(
        schema_version=schema_version,
        source=source,
        channel=channel,
        ts=_utc(provider_ts),
        payload=payload,
    )
    return event


def dedup_fallback_key(event: MarketEventV1) -> str:
    body = "|".join(
        [
            event.source,
            str(event.payload.get("symbol", "")),
            str(event.payload.get("event_type", "")),
            str(event.payload.get("provider_ts", "")),
            hash_payload(event.payload),
        ]
    )
    return sha256(body.encode("utf-8")).hexdigest()


def watermark_late_tag(event: MarketEventV1, watermark: dt.datetime | None) -> dict[str, Any]:
    provider_ts = dt.datetime.fromisoformat(
        str(event.payload["provider_ts"]).replace("Z", "+00:00")
    )
    late = False
    if watermark is not None:
        wm = (
            watermark if watermark.tzinfo is not None else watermark.replace(tzinfo=dt.timezone.utc)
        )
        if provider_ts < wm - dt.timedelta(seconds=10):
            late = True
    return {"late": late}
