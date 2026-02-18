from __future__ import annotations

import json
from typing import Any


class StreamPublisher:
    def __init__(self, redis_client: Any, *, stream_prefix: str = "market") -> None:
        self.redis = redis_client
        self.stream_prefix = stream_prefix

    def publish(self, event: dict[str, Any]) -> str:
        stream = f"{self.stream_prefix}:{event.get('event_type', 'unknown')}"
        return self.redis.xadd(
            stream,
            {
                "event_id": str(event["event_id"]),
                "source": str(event["source"]),
                "symbol": str(event.get("symbol", "")),
                "provider_ts": str(event.get("provider_ts", "")),
                "payload_hash": str(event.get("payload_hash", "")),
                "payload": json.dumps(
                    event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            },
            maxlen=100_000,
            approximate=True,
        )
