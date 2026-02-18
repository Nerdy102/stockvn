from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class _SeriesKey:
    name: str
    labels: tuple[tuple[str, str], ...]


class _MetricRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[_SeriesKey, float] = defaultdict(float)
        self._gauges: dict[_SeriesKey, float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        k = _SeriesKey(name=name, labels=tuple(sorted(labels.items())))
        with self._lock:
            self._counters[k] += float(value)

    def set(self, name: str, value: float, **labels: str) -> None:
        k = _SeriesKey(name=name, labels=tuple(sorted(labels.items())))
        with self._lock:
            self._gauges[k] = float(value)

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        lines: list[str] = []
        for k, v in sorted(counters.items(), key=lambda x: (x[0].name, x[0].labels)):
            lines.append(_fmt_line(k.name, k.labels, v))
        for k, v in sorted(gauges.items(), key=lambda x: (x[0].name, x[0].labels)):
            lines.append(_fmt_line(k.name, k.labels, v))
        return "\n".join(lines) + "\n"


REGISTRY = _MetricRegistry()


def _fmt_line(name: str, labels: tuple[tuple[str, str], ...], value: float) -> str:
    if not labels:
        return f"{name} {value}"
    body = ",".join(f'{k}="{v}"' for k, v in labels)
    return f"{name}{{{body}}} {value}"


class _CounterHandle:
    def __init__(self, name: str, labels: dict[str, str] | None = None) -> None:
        self._name = name
        self._labels = labels or {}

    def labels(self, **labels: str) -> "_CounterHandle":
        merged = dict(self._labels)
        merged.update(labels)
        return _CounterHandle(self._name, merged)

    def inc(self, value: float = 1.0) -> None:
        REGISTRY.inc(self._name, value=value, **self._labels)


class _GaugeHandle:
    def __init__(self, name: str, labels: dict[str, str] | None = None) -> None:
        self._name = name
        self._labels = labels or {}

    def labels(self, **labels: str) -> "_GaugeHandle":
        merged = dict(self._labels)
        merged.update(labels)
        return _GaugeHandle(self._name, merged)

    def set(self, value: float) -> None:
        REGISTRY.set(self._name, value=value, **self._labels)


INGEST_ROWS_TOTAL = _CounterHandle("ingest_rows_total")
INGEST_ERRORS_TOTAL = _CounterHandle("ingest_errors_total")
UPSERT_ROWS_TOTAL = _CounterHandle("upsert_rows_total")

REDIS_STREAM_LAG = _GaugeHandle("redis_stream_lag")
LAST_INGEST_TS = _GaugeHandle("last_ingest_ts")
LAST_FEATURE_TS = _GaugeHandle("last_feature_ts")
LAST_TRAIN_TS = _GaugeHandle("last_train_ts")

BARS_BUILT_TOTAL = _CounterHandle("bars_built_total")
BARS_FINALIZED_TOTAL = _CounterHandle("bars_finalized_total")
LATE_EVENTS_TOTAL = _CounterHandle("late_events_total")
CORRECTIONS_EMITTED_TOTAL = _CounterHandle("corrections_emitted_total")


def mark_now(gauge: _GaugeHandle) -> None:
    gauge.set(time.time())


def metrics_payload() -> tuple[bytes, str]:
    return REGISTRY.render().encode("utf-8"), "text/plain; version=0.0.4"


# initialize metric names for discoverability
REGISTRY.inc("ingest_rows_total", 0.0, channel="bootstrap")
REGISTRY.inc("ingest_errors_total", 0.0, type="bootstrap")
REGISTRY.inc("upsert_rows_total", 0.0, table="bootstrap")
REGISTRY.set("redis_stream_lag", 0.0)
REGISTRY.set("last_ingest_ts", 0.0)
REGISTRY.set("last_feature_ts", 0.0)
REGISTRY.set("last_train_ts", 0.0)
REGISTRY.inc("bars_built_total", 0.0)
REGISTRY.inc("bars_finalized_total", 0.0)
REGISTRY.inc("late_events_total", 0.0)
REGISTRY.inc("corrections_emitted_total", 0.0)


def start_metrics_http_server(port: int = 9001) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore[override]
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            payload, ctype = metrics_payload()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
