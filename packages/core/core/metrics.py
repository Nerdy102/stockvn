from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Dict


class InMemoryMetrics:
    def __init__(self) -> None:
        self._counter: Counter[str] = Counter()
        self._lock = Lock()

    def inc(self, name: str, value: int = 1, **labels: str) -> None:
        key = self._fmt(name, labels)
        with self._lock:
            self._counter[key] += value

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counter)

    @staticmethod
    def _fmt(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        body = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{body}"


METRICS = InMemoryMetrics()
