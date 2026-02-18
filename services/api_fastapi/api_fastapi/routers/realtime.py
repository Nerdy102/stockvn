from __future__ import annotations

import json
from typing import Any

from core.settings import Settings, get_settings
from fastapi import APIRouter, Depends, Query

router = APIRouter(prefix="/realtime", tags=["realtime"])


class _DisabledRealtime:
    pass


def _disabled_payload(message: str) -> dict[str, Any]:
    return {"realtime_disabled": True, "message": message}


def _create_redis_client(redis_url: str):
    from redis import Redis

    return Redis.from_url(redis_url, decode_responses=True)


def get_realtime_client(settings: Settings = Depends(get_settings)) -> Any:
    if not settings.REALTIME_ENABLED:
        return _DisabledRealtime()
    try:
        return _create_redis_client(settings.REDIS_URL)
    except Exception:
        return _DisabledRealtime()


def _safe_json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def _redis_get(redis_client: Any, key: str) -> Any:
    if hasattr(redis_client, "get"):
        return redis_client.get(key)
    return getattr(redis_client, "_kv", {}).get(key)


def _redis_list(redis_client: Any, key: str, limit: int) -> list[Any]:
    if hasattr(redis_client, "lrange"):
        return redis_client.lrange(key, -limit, -1)
    if key.startswith("realtime:bars:"):
        arr = list(getattr(redis_client, "_bar_cache", {}).get(key, []))
    else:
        arr = list(getattr(redis_client, "_hot", {}).get(key, []))
    return arr[-limit:]


def _redis_is_disabled(redis_client: Any) -> bool:
    return isinstance(redis_client, _DisabledRealtime)


@router.get("/summary")
def get_realtime_summary(redis_client: Any = Depends(get_realtime_client)) -> dict[str, Any]:
    if _redis_is_disabled(redis_client):
        return _disabled_payload("Realtime is disabled or unavailable.")
    payload = _safe_json_loads(_redis_get(redis_client, "realtime:ops:summary"), default={})
    if not isinstance(payload, dict):
        return {}
    return payload


@router.get("/bars")
def get_realtime_bars(
    symbol: str = Query(..., min_length=1),
    tf: str = Query("15m"),
    limit: int = Query(default=200, ge=1, le=500),
    redis_client: Any = Depends(get_realtime_client),
) -> dict[str, Any]:
    if _redis_is_disabled(redis_client):
        return _disabled_payload("Realtime is disabled or unavailable.")
    key = f"realtime:bars:{symbol}:{tf}"
    rows = [_safe_json_loads(x, default=x) for x in _redis_list(redis_client, key, limit)]
    return {"symbol": symbol, "tf": tf, "rows": rows}


@router.get("/signals")
def get_realtime_signals(
    symbol: str = Query(..., min_length=1),
    tf: str = Query("15m"),
    limit: int = Query(default=200, ge=1, le=500),
    redis_client: Any = Depends(get_realtime_client),
) -> dict[str, Any]:
    if _redis_is_disabled(redis_client):
        return _disabled_payload("Realtime is disabled or unavailable.")
    key = f"realtime:signals:{symbol}:{tf}"
    raw = _redis_get(redis_client, key)
    payload = _safe_json_loads(raw, default={})
    if isinstance(payload, list):
        payload = payload[-limit:]
    return {"symbol": symbol, "tf": tf, "rows": payload if isinstance(payload, list) else [payload]}


@router.get("/hot/top_movers")
def get_hot_top_movers(
    tf: str = Query("15m"),
    limit: int = Query(default=25, ge=1, le=100),
    redis_client: Any = Depends(get_realtime_client),
) -> dict[str, Any]:
    del tf
    if _redis_is_disabled(redis_client):
        return _disabled_payload("Realtime is disabled or unavailable.")
    rows = [_safe_json_loads(x, default=x) for x in _redis_list(redis_client, "realtime:hot:top_movers", limit)]
    return {"rows": rows}


@router.get("/hot/volume_spikes")
def get_hot_volume_spikes(
    tf: str = Query("15m"),
    limit: int = Query(default=25, ge=1, le=100),
    redis_client: Any = Depends(get_realtime_client),
) -> dict[str, Any]:
    del tf
    if _redis_is_disabled(redis_client):
        return _disabled_payload("Realtime is disabled or unavailable.")
    rows = [
        _safe_json_loads(x, default=x)
        for x in _redis_list(redis_client, "realtime:hot:volume_spike", limit)
    ]
    return {"rows": rows}
