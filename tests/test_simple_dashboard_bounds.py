from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_dashboard_limit_signals_bound() -> None:
    client = TestClient(app)
    r = client.get("/simple/dashboard", params={"limit_signals": 50})
    assert r.status_code == 422


def test_dashboard_lookback_bound() -> None:
    client = TestClient(app)
    r = client.get("/simple/dashboard", params={"lookback_sessions": 1000})
    assert r.status_code == 422
