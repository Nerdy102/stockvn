from __future__ import annotations

import datetime as dt

from core.monitoring.data_health import compute_incident_sla_gauges


def test_sla_gauges_compute_correct() -> None:
    now = dt.datetime(2025, 1, 10, 12, 0, 0)
    incidents = [
        {"status": "OPEN", "created_at": dt.datetime(2025, 1, 10, 6, 0, 0)},   # 6h
        {"status": "OPEN", "created_at": dt.datetime(2025, 1, 9, 6, 0, 0)},    # 30h
        {"status": "OPEN", "created_at": dt.datetime(2025, 1, 6, 0, 0, 0)},    # 108h
        {"status": "CLOSED", "created_at": dt.datetime(2025, 1, 1, 0, 0, 0)},
    ]
    out = compute_incident_sla_gauges(incidents, now=now)
    assert out["open_count"] == 3
    assert out["breach_24h"] == 2
    assert out["breach_72h"] == 1
    assert round(float(out["pct_breach_24h"]), 4) == round(2 / 3, 4)
    assert round(float(out["pct_breach_72h"]), 4) == round(1 / 3, 4)
