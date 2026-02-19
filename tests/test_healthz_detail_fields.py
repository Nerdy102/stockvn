from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_healthz_detail_fields() -> None:
    c = TestClient(app)
    r = c.get('/healthz/detail')
    assert r.status_code == 200
    body = r.json()
    for k in [
        'db_ok',
        'db_latency_ms',
        'broker_ok',
        'data_freshness_ok',
        'kill_switch_state',
        'drift_pause_state',
        'last_reconcile_ts',
    ]:
        assert k in body
