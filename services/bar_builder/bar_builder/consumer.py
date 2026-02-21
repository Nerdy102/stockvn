from __future__ import annotations

import json
from typing import Any


def _has_stream_read(redis_client: Any) -> bool:
    return (
        hasattr(redis_client, "xread")
        and hasattr(redis_client, "get")
        and hasattr(redis_client, "set")
    )


def _decode_rows(rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, fields in rows:
        payload = fields.get("payload")
        if isinstance(payload, str):
            out.append(json.loads(payload))
        elif isinstance(payload, dict):
            out.append(payload)
    return out


def read_market_events(
    redis_client: Any,
    *,
    stream: str = "stream:market_events",
    cursor_key: str = "cursor:market_events",
    block_ms: int = 1_000,
    count: int = 1_000,
) -> list[dict[str, Any]]:
    if _has_stream_read(redis_client):
        last_id = redis_client.get(cursor_key) or "0-0"
        items = redis_client.xread({stream: last_id}, block=block_ms, count=count)
        if not items:
            return []
        _, rows = items[0]
        decoded = _decode_rows(rows)
        if rows:
            redis_client.set(cursor_key, rows[-1][0])
        return decoded

    rows = redis_client.xrange(stream)
    return _decode_rows(rows)
