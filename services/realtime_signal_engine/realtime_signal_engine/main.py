from __future__ import annotations

from typing import Any

from .engine import RealtimeSignalEngine


def health() -> dict[str, str]:
    return {"status": "ok"}


def config_view(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config)


def run_once(engine: RealtimeSignalEngine) -> dict[str, int]:
    return engine.run_once()


def metrics(engine: RealtimeSignalEngine) -> dict[str, Any]:
    return engine.metrics_view()
