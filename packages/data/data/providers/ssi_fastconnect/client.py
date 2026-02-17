from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import httpx

from .token import SsiTokenManager

_RETRY_DELAYS = [0.5, 1.0, 2.0, 4.0, 8.0]
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class SsiRestClient:
    def __init__(
        self,
        *,
        base_url: str,
        token_manager: SsiTokenManager,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_manager = token_manager
        self.timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=30.0)
        self._client = client
        self._sem = asyncio.Semaphore(5)

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        auth_required: bool = True,
    ) -> Any:
        endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        for idx, delay in enumerate([0.0] + _RETRY_DELAYS):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                body = await self._request_once(method, endpoint_path, params, auth_required)
                return body
            except _UnauthorizedError:
                if not auth_required:
                    raise
                self.token_manager.invalidate()
                if idx >= len(_RETRY_DELAYS):
                    raise RuntimeError(f"SSI request unauthorized after refresh: {endpoint_path}")
                continue
            except (httpx.TimeoutException, _RetryableStatusError):
                if idx >= len(_RETRY_DELAYS):
                    raise
                continue

        raise RuntimeError(f"SSI retry attempts exhausted for {endpoint_path}")

    async def _request_once(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None,
        auth_required: bool,
    ) -> Any:
        headers: dict[str, str] = {}
        if auth_required:
            token = await self.token_manager.get_token()
            headers["Authorization"] = f"Bearer {token}"

        async with self._sem:
            if self._client is not None:
                resp = await self._client.request(method, endpoint, params=params, headers=headers)
            else:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                    resp = await client.request(method, endpoint, params=params, headers=headers)

        if resp.status_code == 401:
            raise _UnauthorizedError
        if resp.status_code in _RETRY_STATUS:
            raise _RetryableStatusError(f"retryable status={resp.status_code}")
        resp.raise_for_status()
        return resp.json()


class _RetryableStatusError(RuntimeError):
    pass


class _UnauthorizedError(RuntimeError):
    pass


def fmt_date(v: dt.date | dt.datetime | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v.date().strftime("%d/%m/%Y")
    return v.strftime("%d/%m/%Y")
