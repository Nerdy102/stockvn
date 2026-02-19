from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_simple_dashboard_payload_smoke() -> None:
    client = TestClient(app)
    r = client.get("/simple/dashboard", params={"universe": "VN30", "timeframe": "1D"})
    assert r.status_code == 200
    body = r.json()
    assert "as_of_date" in body
    assert "disclaimers" in body and isinstance(body["disclaimers"], list)
    assert "market_summary" in body
    assert "buy_candidates" in body and "sell_candidates" in body
    assert "model_leaderboard" in body
