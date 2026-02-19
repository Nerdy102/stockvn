from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_api_simple_mode_bounds_limit() -> None:
    client = TestClient(app)
    r = client.post(
        "/simple/run_compare", json={"symbols": [f"S{i}" for i in range(60)], "lookback_days": 252}
    )
    assert r.status_code == 422
