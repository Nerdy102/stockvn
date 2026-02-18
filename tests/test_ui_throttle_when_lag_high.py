from __future__ import annotations

from apps.dashboard_streamlit.ui.realtime_poll import (
    DEFAULT_POLL_SECONDS,
    THROTTLED_POLL_SECONDS,
    compute_poll_interval_seconds,
)


def test_ui_throttle_when_lag_high() -> None:
    assert compute_poll_interval_seconds(1.0) == DEFAULT_POLL_SECONDS
    assert compute_poll_interval_seconds(5.0) == DEFAULT_POLL_SECONDS
    assert compute_poll_interval_seconds(5.01) == THROTTLED_POLL_SECONDS
