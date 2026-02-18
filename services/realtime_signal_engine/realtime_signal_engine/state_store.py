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

    def get_indicator_state(self, symbol: str, tf: str) -> IndicatorState | None:
        raw = self.redis._kv.get(self._k_ind(symbol, tf))
        if raw is None:
            return None
        return IndicatorState.from_json(json.loads(raw))

    def set_indicator_state(self, symbol: str, tf: str, state: IndicatorState) -> None:
        self.redis._kv[self._k_ind(symbol, tf)] = json.dumps(
            state.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def set_signal_snapshot(self, symbol: str, tf: str, payload: dict[str, Any]) -> None:
        self.redis._kv[self._k_signals(symbol, tf)] = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def get_hot_cache_bars(self, symbol: str, tf: str) -> list[dict[str, Any]]:
        arr = getattr(self.redis, "_bar_cache", {}).get(f"realtime:bars:{symbol}:{tf}", [])
        out: list[dict[str, Any]] = []
        for s in arr[-200:]:
            if isinstance(s, str):
                out.append(json.loads(s))
            elif isinstance(s, dict):
                out.append(s)
        return out

    def set_ops_summary(self, payload: dict[str, Any]) -> None:
        self.redis._kv["realtime:ops:summary"] = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def push_hot(self, name: str, payload: dict[str, Any], limit: int = 50) -> None:
        key = f"realtime:hot:{name}"
        cur = getattr(self.redis, "_hot", {}).get(key, [])
        if not hasattr(self.redis, "_hot"):
            self.redis._hot = {}
        cur.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        self.redis._hot[key] = cur[-limit:]
