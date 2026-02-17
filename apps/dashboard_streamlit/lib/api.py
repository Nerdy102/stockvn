from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


def _base() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
    url = _base() + path
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def post(path: str, json: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> Any:
    url = _base() + path
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=json)
        r.raise_for_status()
        return r.json()
