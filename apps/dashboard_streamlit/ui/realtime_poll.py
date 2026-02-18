from __future__ import annotations

import time
from typing import Any

import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json

SUMMARY_PATH = "/realtime/summary"
DEFAULT_POLL_SECONDS = 2.0
THROTTLED_POLL_SECONDS = 5.0
LAG_THROTTLE_THRESHOLD_S = 5.0


def compute_poll_interval_seconds(stream_lag_s: float | int | None) -> float:
    if stream_lag_s is None:
        return DEFAULT_POLL_SECONDS
    try:
        lag = float(stream_lag_s)
    except Exception:
        return DEFAULT_POLL_SECONDS
    return THROTTLED_POLL_SECONDS if lag > LAG_THROTTLE_THRESHOLD_S else DEFAULT_POLL_SECONDS


def is_polling_enabled(ui_mode: str, realtime_enabled: bool) -> bool:
    if not realtime_enabled:
        return False
    return str(ui_mode).upper() in {"LIVE", "PAPER"}


def poll_summary(
    *,
    ui_mode: str,
    realtime_enabled: bool,
    now_ts: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    base_state: dict[str, Any] = {
        "enabled": is_polling_enabled(ui_mode, realtime_enabled),
        "interval_s": DEFAULT_POLL_SECONDS,
        "throttled": False,
        "stream_lag_s": None,
        "realtime_disabled": not realtime_enabled,
    }
    if not base_state["enabled"]:
        return base_state

    now = time.time() if now_ts is None else float(now_ts)
    state = st.session_state.setdefault(
        "realtime_poll",
        {"_payload": None, "_last_ts": 0.0, "_interval_s": DEFAULT_POLL_SECONDS},
    )
    last_payload = state.get("_payload")
    last_ts = state.get("_last_ts", 0.0)
    interval_s = state.get("_interval_s", DEFAULT_POLL_SECONDS)

    if not force and (now - float(last_ts)) < float(interval_s) and isinstance(last_payload, dict):
        return last_payload

    try:
        payload = cached_get_json(SUMMARY_PATH, params=None, ttl_s=1)
        if not isinstance(payload, dict):
            payload = {"message": "invalid realtime summary payload"}
    except Exception:
        payload = {"realtime_disabled": True, "message": "Realtime summary unavailable."}

    lag = payload.get("stream_lag_s")
    next_interval = compute_poll_interval_seconds(lag)
    payload["interval_s"] = next_interval
    payload["throttled"] = next_interval >= THROTTLED_POLL_SECONDS
    payload["enabled"] = True

    state["_payload"] = payload
    state["_last_ts"] = now
    state["_interval_s"] = next_interval
    return payload
