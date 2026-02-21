from __future__ import annotations

import json
from typing import Any

from indicators.incremental import IndicatorState


class StateStore:
    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    def _k_ind(self, symbol: str, tf: str) -> str:
        return f"realtime:state:ind:{symbol}:{tf}"

    def _k_signals(self, symbol: str, tf: str) -> str:
        return f"realtime:signals:{symbol}:{tf}"

    def get_indicator_state(self, symbol: str, tf: str) -> IndicatorState | None:
        raw = (
            self.redis.get(self._k_ind(symbol, tf))
            if hasattr(self.redis, "get")
            else getattr(self.redis, "_kv", {}).get(self._k_ind(symbol, tf))
        )
        if raw is None:
            return None
        return IndicatorState.from_json(json.loads(raw))

    def set_indicator_state(self, symbol: str, tf: str, state: IndicatorState) -> None:
        payload = json.dumps(
            state.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if hasattr(self.redis, "set"):
            self.redis.set(self._k_ind(symbol, tf), payload)
        else:
            if not hasattr(self.redis, "_kv"):
                self.redis._kv = {}
            self.redis._kv[self._k_ind(symbol, tf)] = payload

    def set_signal_snapshot(self, symbol: str, tf: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hasattr(self.redis, "set"):
            self.redis.set(self._k_signals(symbol, tf), encoded)
        else:
            if not hasattr(self.redis, "_kv"):
                self.redis._kv = {}
            self.redis._kv[self._k_signals(symbol, tf)] = encoded

    def get_hot_cache_bars(self, symbol: str, tf: str) -> list[dict[str, Any]]:
        key = f"realtime:bars:{symbol}:{tf}"
        arr = (
            self.redis.lrange(key, -200, -1)
            if hasattr(self.redis, "lrange")
            else getattr(self.redis, "_bar_cache", {}).get(key, [])
        )
        out: list[dict[str, Any]] = []
        for s in arr[-200:]:
            out.append(json.loads(s) if isinstance(s, str) else s)
        return out

    def set_ops_summary(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hasattr(self.redis, "set"):
            self.redis.set("realtime:ops:summary", encoded)
        else:
            if not hasattr(self.redis, "_kv"):
                self.redis._kv = {}
            self.redis._kv["realtime:ops:summary"] = encoded

    def push_hot(self, name: str, payload: dict[str, Any], limit: int = 50) -> None:
        key = f"realtime:hot:{name}"
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hasattr(self.redis, "rpush") and hasattr(self.redis, "ltrim"):
            self.redis.rpush(key, encoded)
            self.redis.ltrim(key, -limit, -1)
            return
        if not hasattr(self.redis, "_hot"):
            self.redis._hot = {}
        cur = list(self.redis._hot.get(key, []))
        cur.append(encoded)
        self.redis._hot[key] = cur[-limit:]
