from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_kiosk_limit_signals_bound() -> None:
    client = TestClient(app)
    r = client.get("/simple/kiosk", params={"limit_signals": 99})
    assert r.status_code == 422
