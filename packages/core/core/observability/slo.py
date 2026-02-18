from __future__ import annotations

import datetime as dt
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WINDOW_SECONDS = 300


@dataclass(frozen=True)
class IncidentRule:
    code: str
    metric: str
    threshold: float
    severity: str
    runbook_id: str
    min_window_s: int = WINDOW_SECONDS


INCIDENT_RULES: tuple[IncidentRule, ...] = (
    IncidentRule(
        code="REALTIME_LAG_HIGH",
        metric="ingest_lag_s_p95",
        threshold=5.0,
        severity="HIGH",
        runbook_id="runbook:realtime_lag",
    ),
    IncidentRule(
        code="BAR_BUILD_SLOW",
        metric="bar_build_latency_s_p95",
        threshold=3.0,
        severity="HIGH",
        runbook_id="runbook:bar_perf",
    ),
    IncidentRule(
        code="SIGNAL_SLOW",
        metric="signal_latency_s_p95",
        threshold=5.0,
        severity="MEDIUM",
        runbook_id="runbook:signal_perf",
    ),
    IncidentRule(
        code="STREAM_BACKLOG",
        metric="redis_stream_pending",
        threshold=50_000.0,
        severity="HIGH",
        runbook_id="runbook:redis_backlog",
        min_window_s=0,
    ),
)


class RollingSLO:
    def __init__(self, window_seconds: int = WINDOW_SECONDS) -> None:
        self.window_seconds = int(window_seconds)
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, value: float, ts: dt.datetime | float | None = None) -> None:
        if ts is None:
            t = dt.datetime.utcnow().timestamp()
        elif isinstance(ts, dt.datetime):
            t = ts.timestamp()
        else:
            t = float(ts)
        self._samples.append((t, float(value)))
        self._trim(now_ts=t)

    def _trim(self, now_ts: float) -> None:
        cutoff = now_ts - float(self.window_seconds)
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def snapshot(self, now: dt.datetime | float | None = None) -> dict[str, Any]:
        now_ts = (
            dt.datetime.utcnow().timestamp()
            if now is None
            else (now.timestamp() if isinstance(now, dt.datetime) else float(now))
        )
        self._trim(now_ts=now_ts)
        arr = np.asarray([v for _, v in self._samples], dtype=float)
        if arr.size == 0:
            return {
                "count": 0,
                "window_s": self.window_seconds,
                "p50": 0.0,
                "p95": 0.0,
                "last": 0.0,
                "max": 0.0,
            }
        return {
            "count": int(arr.size),
            "window_s": self.window_seconds,
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "last": float(arr[-1]),
            "max": float(np.max(arr)),
        }


def build_slo_snapshot(
    *,
    service: str,
    ingest_lag: dict[str, Any] | None = None,
    bar_latency: dict[str, Any] | None = None,
    signal_latency: dict[str, Any] | None = None,
    redis_stream_pending: float | int = 0.0,
) -> dict[str, Any]:
    ingest_lag = ingest_lag or {}
    bar_latency = bar_latency or {}
    signal_latency = signal_latency or {}
    return {
        "service": service,
        "as_of": dt.datetime.utcnow().isoformat() + "Z",
        "window_s": int(
            max(
                ingest_lag.get("window_s", WINDOW_SECONDS),
                bar_latency.get("window_s", WINDOW_SECONDS),
                signal_latency.get("window_s", WINDOW_SECONDS),
            )
        ),
        "ingest_lag_s_p50": float(ingest_lag.get("p50", 0.0)),
        "ingest_lag_s_p95": float(ingest_lag.get("p95", 0.0)),
        "bar_build_latency_s_p50": float(bar_latency.get("p50", 0.0)),
        "bar_build_latency_s_p95": float(bar_latency.get("p95", 0.0)),
        "signal_latency_s_p50": float(signal_latency.get("p50", 0.0)),
        "signal_latency_s_p95": float(signal_latency.get("p95", 0.0)),
        "redis_stream_pending": float(redis_stream_pending),
    }


def snapshot_to_metrics_json(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "slo": snapshot}


def save_snapshot(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def load_snapshots(paths: list[str | Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out.append(data)
        except Exception:
            continue
    return out


def evaluate_incidents_from_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for snap in snapshots:
        service = str(snap.get("service", "unknown"))
        window_s = int(snap.get("window_s", WINDOW_SECONDS))
        for rule in INCIDENT_RULES:
            if rule.min_window_s > 0 and window_s < rule.min_window_s:
                continue
            value = float(snap.get(rule.metric, 0.0))
            if value > rule.threshold:
                incidents.append(
                    {
                        "source": f"realtime_ops:{service}",
                        "severity": rule.severity,
                        "status": "OPEN",
                        "summary": f"{rule.code} on {service}",
                        "runbook_section": rule.runbook_id,
                        "details_json": {
                            "code": rule.code,
                            "service": service,
                            "metric": rule.metric,
                            "threshold": rule.threshold,
                            "value": value,
                            "window_s": window_s,
                        },
                    }
                )
    return incidents
