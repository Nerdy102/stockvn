from __future__ import annotations

import asyncio
import json
from pathlib import Path

import websockets
from helpers_redis_fake import FakeAsyncRedisCompat

FIX = Path("tests/fixtures/ssi_streaming")


class DummyTokenManager:
    async def get_token(self) -> str:
        return "dummy"


def _build_async_redis():
    try:
        import fakeredis.aioredis as fakeredis_aioredis  # type: ignore

        return fakeredis_aioredis.FakeRedis(decode_responses=True)
    except Exception:
        return FakeAsyncRedisCompat()


def test_stream_ingestor_writes_to_redis_stream() -> None:
    from data.providers.ssi_fastconnect.provider_stream import SsiStreamIngestor

    async def run() -> None:
        payload = (FIX / "msg_X.json").read_text(encoding="utf-8")

        async def handler(ws) -> None:
            await ws.recv()
            await ws.send(payload)
            await ws.close()

        ingestor = SsiStreamIngestor(
            stream_url="ws://127.0.0.1:9876",
            redis_url="redis://unused",
            token_manager=DummyTokenManager(),
            subscribe_universe="VN30,VNINDEX",
        )
        ingestor.redis_client = _build_async_redis()

        async with websockets.serve(handler, "127.0.0.1", 9876):
            await ingestor._run_once()

        rows = await ingestor.redis_client.xrange("ssi:X")
        assert len(rows) == 1
        _, fields = rows[0]
        assert fields["rtype"] == "X"
        body = json.loads(fields["payload"])
        assert body["DataType"] == "X"

    asyncio.run(run())
