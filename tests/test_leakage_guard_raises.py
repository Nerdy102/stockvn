from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from core.alpha_v3.features import assert_no_leakage, enforce_no_leakage_guard


def test_assert_no_leakage_raises_on_overlap() -> None:
    with pytest.raises(RuntimeError):
        assert_no_leakage(dt.date(2024, 1, 1), dt.date(2024, 1, 20), horizon=21)


def test_enforce_no_leakage_guard_raises_with_future_injection() -> None:
    bad = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "label_date": [dt.date(2024, 1, 30), dt.date(2024, 1, 10)],
        }
    )
    with pytest.raises(RuntimeError):
        enforce_no_leakage_guard(bad, horizon=21)


def test_assert_no_leakage_allows_exact_horizon_boundary() -> None:
    # 21 trading days from 2024-01-01 lands on 2024-01-31 with VN holiday/weekend skips.
    assert_no_leakage(dt.date(2024, 1, 1), dt.date(2024, 1, 31), horizon=21)
