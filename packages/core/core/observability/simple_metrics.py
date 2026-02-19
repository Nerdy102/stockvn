from __future__ import annotations

import math
import time
from collections import defaultdict

_state = {
    "orders_created_total": 0,
    "orders_executed_total": 0,
    "orders_blocked_total": defaultdict(int),
    "broker_errors_total": 0,
    "latencies_ms": [],
}


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def observe_api_latency(latency_ms: float) -> None:
    arr = _state["latencies_ms"]
    arr.append(float(latency_ms))
    if len(arr) > 2000:
        del arr[: len(arr) - 2000]


def inc_created() -> None:
    _state["orders_created_total"] += 1


def inc_executed() -> None:
    _state["orders_executed_total"] += 1


def inc_blocked(reason_code: str) -> None:
    _state["orders_blocked_total"][str(reason_code or "UNKNOWN")] += 1


def inc_broker_error() -> None:
    _state["broker_errors_total"] += 1


def snapshot() -> dict:
    lat = sorted(_state["latencies_ms"])
    if not lat:
        p95 = 0.0
    else:
        idx = max(0, min(len(lat) - 1, math.ceil(len(lat) * 0.95) - 1))
        p95 = float(lat[idx])
    return {
        "orders_created_total": int(_state["orders_created_total"]),
        "orders_executed_total": int(_state["orders_executed_total"]),
        "orders_blocked_total": dict(_state["orders_blocked_total"]),
        "api_latency_p95_ms": p95,
        "broker_errors_total": int(_state["broker_errors_total"]),
    }
