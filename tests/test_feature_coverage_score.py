from __future__ import annotations

import datetime as dt

import pandas as pd

from worker_scheduler.jobs import _coverage_metrics_for_date


def test_coverage_score_decreases_when_missing_injected() -> None:
    cols = ["ret_21d", "vol_20d", "adv20_value", "rsi14"]
    complete = pd.DataFrame(
        [
            {"symbol": "AAA", "date": dt.date(2024, 1, 2), "ret_21d": 0.1, "vol_20d": 0.2, "adv20_value": 100.0, "rsi14": 51.0},
            {"symbol": "BBB", "date": dt.date(2024, 1, 2), "ret_21d": 0.2, "vol_20d": 0.3, "adv20_value": 120.0, "rsi14": 55.0},
        ]
    )
    missing = complete.copy()
    missing.loc[0, "ret_21d"] = None
    missing.loc[1, "vol_20d"] = None

    score_complete, _ = _coverage_metrics_for_date(complete, cols)
    score_missing, metrics_missing = _coverage_metrics_for_date(missing, cols)

    assert score_missing < score_complete
    assert metrics_missing["symbols_dropped_pct"] > 0.0
