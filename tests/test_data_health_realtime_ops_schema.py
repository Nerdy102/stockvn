from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_data_health_realtime_ops_schema() -> None:
    c = TestClient(create_app())
    r = c.get("/data/health/realtime_ops")
    assert r.status_code == 200
    body = r.json()
    assert "gauges" in body
    assert "incidents" in body
    assert "runbooks" in body

    gauges = body["gauges"]
    for k in [
        "ingest_lag_s_p95",
        "bar_build_latency_s_p95",
        "signal_latency_s_p95",
        "redis_stream_pending",
    ]:
        assert k in gauges

    runbooks = body["runbooks"]
    assert runbooks["REALTIME_LAG_HIGH"] == "runbook:realtime_lag"
    assert runbooks["STREAM_BACKLOG"] == "runbook:redis_backlog"
