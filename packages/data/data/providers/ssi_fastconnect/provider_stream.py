from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import websockets
from core.logging import get_logger

from data.providers.ssi_fastconnect.mapper_stream import normalize_rtype, parse_content
from data.providers.ssi_fastconnect.token import SsiTokenManager

try:
    import redis.asyncio as redis_async
except Exception:  # pragma: no cover
    redis_async = None

log = get_logger(__name__)


class SsiStreamIngestor:
    def __init__(
        self,
        *,
        stream_url: str,
        redis_url: str,
        token_manager: SsiTokenManager,
        subscribe_universe: str = "VN30,VNINDEX",
        ping_interval_s: int = 20,
        idle_timeout_s: int = 60,
        stream_maxlen: int = 2_000_000,
    ) -> None:
        self.stream_url = stream_url
        self.redis_client = (
            redis_async.from_url(redis_url, decode_responses=True) if redis_async is not None else None
        )
        self.token_manager = token_manager
        self.subscribe_universe = subscribe_universe
        self.ping_interval_s = ping_interval_s
        self.idle_timeout_s = idle_timeout_s
        self.stream_maxlen = stream_maxlen
        self._last_recv_monotonic = time.monotonic()

    async def run_forever(self) -> None:
        backoff_seq = [1, 2, 4, 8, 16, 32]
        attempts = 0
        while True:
            try:
                await self._run_once()
                attempts = 0
            except Exception as exc:
                wait_s = min(backoff_seq[min(attempts, len(backoff_seq) - 1)], 60)
                attempts += 1
                log.warning("ssi_stream_reconnect", extra={"error": str(exc), "backoff_s": wait_s})
                await asyncio.sleep(wait_s)

    async def _run_once(self) -> None:
        token = await self.token_manager.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Access-Token": token,
        }
        async with websockets.connect(
            self.stream_url,
            additional_headers=headers,
            ping_interval=None,
            close_timeout=5,
        ) as ws:
            self._last_recv_monotonic = time.monotonic()
            for msg in self._build_subscribe_messages(token):
                await ws.send(json.dumps(msg, ensure_ascii=False))
            hb_task = asyncio.create_task(self._heartbeat(ws))
            try:
                async for raw in ws:
                    self._last_recv_monotonic = time.monotonic()
                    if isinstance(raw, str):
                        await self._handle_ws_message(raw)
            finally:
                hb_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await hb_task

    def _build_subscribe_messages(self, token: str) -> list[dict[str, Any]]:
        symbols = [s.strip().upper() for s in self.subscribe_universe.split(",") if s.strip()]
        if not symbols:
            symbols = ["VN30", "VNINDEX"]

        # SSI stream endpoints commonly support channel-wise subscribe payloads.
        # Keep message explicit per channel to avoid sending an "ALL" wildcard.
        channels = ["X", "X-QUOTE", "X-TRADE", "R", "MI", "B", "F", "OL"]
        return [
            {
                "type": "subscribe",
                "channel": channel,
                "symbols": symbols,
                "accessToken": token,
            }
            for channel in channels
        ]

    async def _heartbeat(self, ws: websockets.ClientConnection) -> None:
        while True:
            await asyncio.sleep(self.ping_interval_s)
            idle = time.monotonic() - self._last_recv_monotonic
            if idle > self.idle_timeout_s:
                await ws.close(code=1011, reason="heartbeat timeout")
                return
            pong = await ws.ping()
            await pong

    async def _handle_ws_message(self, raw: str) -> None:
        payload = json.loads(raw)
        content = parse_content(payload)
        rtype = normalize_rtype(payload, content)
        ts_ms = int(time.time() * 1000)
        if self.redis_client is None:
            raise RuntimeError("redis.asyncio is required for stream ingestion runtime")
        await self.redis_client.xadd(
            f"ssi:{rtype}",
            {
                "ts_recv_utc": str(ts_ms),
                "payload": raw,
                "rtype": rtype,
            },
            maxlen=self.stream_maxlen,
            approximate=True,
        )
