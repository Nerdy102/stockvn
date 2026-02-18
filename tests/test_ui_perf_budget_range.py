from __future__ import annotations

import datetime as dt

import pytest

from apps.dashboard_streamlit.ui.perf import DAILY_MAX_DAYS_DEFAULT, enforce_bounded_range


def test_ui_perf_budget_range_raises_when_over_budget() -> None:
    with pytest.raises(ValueError):
        enforce_bounded_range(dt.date(2023, 1, 1), dt.date(2025, 12, 31), DAILY_MAX_DAYS_DEFAULT)
