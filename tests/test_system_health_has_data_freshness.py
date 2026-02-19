from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_system_health_includes_data_freshness() -> None:
    client = TestClient(app)
    resp = client.get('/simple/system_health')
    assert resp.status_code == 200
    body = resp.json()
    assert 'data_freshness' in body
    assert 'status' in body['data_freshness']
