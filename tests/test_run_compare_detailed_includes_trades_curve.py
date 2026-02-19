from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_run_compare_detailed_includes_trades_curve() -> None:
    client = TestClient(app)
    r = client.post(
        "/simple/run_compare",
        json={
            "symbols": ["FPT"],
            "timeframe": "1D",
            "lookback_days": 252,
            "detail_level": "chi tiết",
            "include_equity_curve": True,
            "include_trades": True,
            "execution": "giá đóng cửa (close)",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "leaderboard" in body and body["leaderboard"]
    first = body["leaderboard"][0]
    assert "equity_curve" in first
    assert "trade_list" in first
