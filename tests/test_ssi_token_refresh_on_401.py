from __future__ import annotations

import asyncio

import httpx

from data.providers.ssi_fastconnect.client import SsiRestClient
from data.providers.ssi_fastconnect.token import SsiTokenManager


class DummyTokenManager(SsiTokenManager):
    def __init__(self) -> None:
        super().__init__(base_url="https://example.test", consumer_id="x", consumer_secret="y", private_key="z")
        self.calls = 0

    async def get_token(self) -> str:
        self.calls += 1
        return "token-a" if self.calls == 1 else "token-b"


def test_ssi_token_refresh_on_401() -> None:
    call_count = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(401, json={"message": "unauthorized"})
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(_handler)
    token_manager = DummyTokenManager()

    async def _run() -> None:
        async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as http_client:
            client = SsiRestClient(
                base_url="https://example.test", token_manager=token_manager, client=http_client
            )
            data = await client.request("GET", "/Securities")
            assert data == []

    asyncio.run(_run())
    assert call_count["n"] == 2
    assert token_manager.calls == 2
