from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_realtime_summary_graceful_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("REALTIME_ENABLED", "false")
    c = TestClient(create_app())
    r = c.get("/realtime/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["realtime_disabled"] is True
    assert "message" in body
