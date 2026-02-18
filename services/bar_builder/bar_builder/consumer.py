from __future__ import annotations

import json
from typing import Any


def read_market_events(
    redis_client: Any, *, stream: str = "stream:market_events"
) -> list[dict[str, Any]]:
    rows = redis_client.xrange(stream)
    out: list[dict[str, Any]] = []
    for _, fields in rows:
        payload = fields.get("payload")
        if isinstance(payload, str):
            out.append(json.loads(payload))
        elif isinstance(payload, dict):
            out.append(payload)
    return out
