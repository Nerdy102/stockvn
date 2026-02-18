from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_signals_rejects_unbounded_request_without_cursor_or_range() -> None:
    c = TestClient(create_app())
    r = c.get("/signals", params={"limit": 100})
    assert r.status_code == 400
    assert "cursor is required" in r.text


def test_fundamentals_rejects_unbounded_request_without_cursor_or_range() -> None:
    c = TestClient(create_app())
    r = c.get("/fundamentals", params={"limit": 100})
    assert r.status_code == 400
    assert "cursor is required" in r.text


def test_prices_rejects_unbounded_request_without_cursor_or_range() -> None:
    c = TestClient(create_app())
    r = c.get("/prices", params={"symbol": "AAA", "timeframe": "1D", "limit": 100})
    assert r.status_code == 400
    assert "cursor is required" in r.text
