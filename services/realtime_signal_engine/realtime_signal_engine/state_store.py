from __future__ import annotations

import json
from typing import Any

from indicators.incremental import IndicatorState


class StateStore:
    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client
        if not hasattr(self.redis, "_kv"):
            self.redis._kv = {}

    def _k_ind(self, symbol: str, tf: str) -> str:
        return f"realtime:state:ind:{symbol}:{tf}"

    def _k_signals(self, symbol: str, tf: str) -> str:
        return f"realtime:signals:{symbol}:{tf}"

    def _supports_kv(self) -> bool:
        return hasattr(self.redis, "get") and hasattr(self.redis, "set")

    def _supports_hot_list(self) -> bool:
        return hasattr(self.redis, "rpush") and hasattr(self.redis, "ltrim")

    def get_indicator_state(self, symbol: str, tf: str) -> IndicatorState | None:
        key = self._k_ind(symbol, tf)
        raw = self.redis.get(key) if self._supports_kv() else self.redis._kv.get(key)
        if raw is None:
            return None
        return IndicatorState.from_json(json.loads(raw))

    def set_indicator_state(self, symbol: str, tf: str, state: IndicatorState) -> None:
        key = self._k_ind(symbol, tf)
        payload = json.dumps(
            state.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if self._supports_kv():
            self.redis.set(key, payload)
        else:
            self.redis._kv[key] = payload

    def set_signal_snapshot(self, symbol: str, tf: str, payload: dict[str, Any]) -> None:
        key = self._k_signals(symbol, tf)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self._supports_kv():
            self.redis.set(key, encoded)
        else:
            self.redis._kv[key] = encoded

    def get_hot_cache_bars(self, symbol: str, tf: str) -> list[dict[str, Any]]:
        key = f"realtime:bars:{symbol}:{tf}"
        if hasattr(self.redis, "lrange"):
            arr = self.redis.lrange(key, -200, -1)
        else:
            arr = getattr(self.redis, "_bar_cache", {}).get(key, [])
        out: list[dict[str, Any]] = []
        for s in arr[-200:]:
            if isinstance(s, str):
                out.append(json.loads(s))
            elif isinstance(s, dict):
                out.append(s)
        return out

    def set_ops_summary(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self._supports_kv():
            self.redis.set("realtime:ops:summary", encoded)
        else:
            self.redis._kv["realtime:ops:summary"] = encoded

    def push_hot(self, name: str, payload: dict[str, Any], limit: int = 50) -> None:
        key = f"realtime:hot:{name}"
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self._supports_hot_list():
            self.redis.rpush(key, encoded)
            self.redis.ltrim(key, -limit, -1)
            return

        cur = getattr(self.redis, "_hot", {}).get(key, [])
        if not hasattr(self.redis, "_hot"):
            self.redis._hot = {}
        cur.append(encoded)
        self.redis._hot[key] = cur[-limit:]
