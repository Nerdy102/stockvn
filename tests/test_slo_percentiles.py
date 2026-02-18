from __future__ import annotations

import datetime as dt

from core.observability.slo import RollingSLO


def test_slo_percentiles_over_rolling_window() -> None:
    s = RollingSLO(window_seconds=300)
    now = dt.datetime(2025, 1, 1, 9, 0, 0)
    for i in range(10):
        s.add(float(i), ts=now + dt.timedelta(seconds=i * 10))

    snap = s.snapshot(now=now + dt.timedelta(seconds=120))
    assert snap["count"] == 10
    assert 4.0 <= snap["p50"] <= 5.0
    assert snap["p95"] >= 8.0

    # older sample falls out of 5m window
    s.add(100.0, ts=now + dt.timedelta(seconds=400))
    snap2 = s.snapshot(now=now + dt.timedelta(seconds=401))
    assert snap2["count"] < 11
    assert snap2["last"] == 100.0
