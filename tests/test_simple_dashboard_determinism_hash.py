from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_dashboard_determinism_hash() -> None:
    client = TestClient(app)
    p = {"universe": "VN30", "timeframe": "1D", "limit_signals": 10, "lookback_sessions": 252}
    r1 = client.get("/simple/dashboard", params=p)
    r2 = client.get("/simple/dashboard", params=p)
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1.get("report_id") == b2.get("report_id")
    assert b1["model_performance_leaderboard"][0]["config_hash"] == b2["model_performance_leaderboard"][0]["config_hash"]
