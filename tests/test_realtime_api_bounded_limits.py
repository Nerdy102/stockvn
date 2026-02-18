from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_realtime_bars_limit_is_bounded() -> None:
    c = TestClient(create_app())
    r = c.get("/realtime/bars", params={"symbol": "AAA", "tf": "15m", "limit": 501})
    assert r.status_code == 422


def test_realtime_hot_limit_is_bounded() -> None:
    c = TestClient(create_app())
    r = c.get("/realtime/hot/top_movers", params={"tf": "15m", "limit": 101})
    assert r.status_code == 422
