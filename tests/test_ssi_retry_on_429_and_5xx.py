from __future__ import annotations

import asyncio

import httpx
import pytest

from data.providers.ssi_fastconnect.client import SsiRestClient
from data.providers.ssi_fastconnect.token import SsiTokenManager


class StaticTokenManager(SsiTokenManager):
    def __init__(self) -> None:
        super().__init__(base_url="https://example.test", consumer_id="x", consumer_secret="y", private_key="z")

    async def get_token(self) -> str:
        return "token"


def test_ssi_retry_on_429_and_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    call_count = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(429, json={"message": "rate"})
        if call_count["n"] == 2:
            return httpx.Response(500, json={"message": "server"})
        return httpx.Response(200, json=[{"ok": 1}])

    transport = httpx.MockTransport(_handler)

    async def _run() -> None:
        async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as http_client:
            client = SsiRestClient(
                base_url="https://example.test",
                token_manager=StaticTokenManager(),
                client=http_client,
            )
            payload = await client.request("GET", "/DailyOhlc")
            assert payload == [{"ok": 1}]

    asyncio.run(_run())
    assert call_count["n"] == 3
    assert sleeps == [0.5, 1.0]
