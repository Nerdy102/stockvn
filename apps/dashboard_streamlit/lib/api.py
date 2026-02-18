from __future__ import annotations

import os
from typing import Any

import httpx


def api_base() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def get(path: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    url = api_base() + path
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def post(path: str, json: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    url = api_base() + path
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=json)
        r.raise_for_status()
        return r.json()


def get_bytes(path: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> bytes:
    url = api_base() + path
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.content
