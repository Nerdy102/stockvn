from __future__ import annotations

import asyncio
import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

import httpx

_RETRY_DELAYS = [0.5, 1.0, 2.0, 4.0, 8.0]
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class _TokenState:
    token: str | None = None
    expires_at_utc: dt.datetime | None = None


class SsiTokenManager:
    """Manage SSI access token with in-memory cache and refresh support."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        consumer_id: str | None = None,
        consumer_secret: str | None = None,
        private_key: str | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("SSI_FCDATA_BASE_URL", "")).rstrip("/")
        self.consumer_id = consumer_id or os.getenv("SSI_CONSUMER_ID", "")
        self.consumer_secret = consumer_secret or os.getenv("SSI_CONSUMER_SECRET", "")
        self.private_key = private_key or os.getenv("SSI_PRIVATE_KEY", "")
        self.timeout = timeout or httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0)
        self._state = _TokenState()
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._state = _TokenState()

    async def get_token(self) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        state = self._state
        if state.token and state.expires_at_utc and state.expires_at_utc > now:
            return state.token

        async with self._lock:
            now = dt.datetime.now(dt.timezone.utc)
            state = self._state
            if state.token and state.expires_at_utc and state.expires_at_utc > now:
                return state.token
            token, expires_at = await self._fetch_token()
            self._state = _TokenState(token=token, expires_at_utc=expires_at)
            return token

    async def _fetch_token(self) -> tuple[str, dt.datetime]:
        if not self.base_url:
            raise RuntimeError("Missing SSI_FCDATA_BASE_URL for REST token endpoint.")
        missing = [
            key
            for key, value in {
                "SSI_CONSUMER_ID": self.consumer_id,
                "SSI_CONSUMER_SECRET": self.consumer_secret,
                "SSI_PRIVATE_KEY": self.private_key,
            }.items()
            if not str(value).strip()
        ]
        if missing:
            raise RuntimeError(f"Missing SSI credentials for token request: {', '.join(missing)}")

        payload = {
            "consumerID": self.consumer_id,
            "consumerSecret": self.consumer_secret,
            "privateKey": self.private_key,
        }
        body: dict[str, Any] | None = None
        async with httpx.AsyncClient(timeout=self.timeout, base_url=self.base_url) as client:
            for idx, delay in enumerate([0.0] + _RETRY_DELAYS):
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    resp = await client.post("/AccessToken", json=payload)
                except httpx.TimeoutException:
                    if idx >= len(_RETRY_DELAYS):
                        raise
                    continue
                if resp.status_code in _RETRY_STATUS:
                    if idx >= len(_RETRY_DELAYS):
                        resp.raise_for_status()
                    continue
                resp.raise_for_status()
                body = resp.json()
                break

        if body is None:
            raise RuntimeError("SSI token request exhausted retries without response body")

        token = body.get("accessToken") or body.get("AccessToken")
        if not token:
            raise RuntimeError(f"SSI token response missing accessToken. Keys={sorted(body.keys())}")

        expires_seconds = _to_int(body.get("expiresIn") or body.get("expires_in"))
        ttl_seconds = expires_seconds if expires_seconds and expires_seconds > 0 else 25 * 60
        expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=ttl_seconds)
        return str(token), expires_at


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except ValueError:
        return None
