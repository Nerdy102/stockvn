from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_run_compare_story_fields_requested() -> None:
    client = TestClient(app)
    r = client.post(
        "/simple/run_compare",
        json={
            "symbols": ["FPT", "VNM", "VCB", "MWG", "HPG"],
            "timeframe": "1D",
            "lookback_days": 252,
            "detail_level": "tóm tắt",
            "execution": "giá đóng cửa (close)",
            "market": "vn",
            "trading_type": "spot_paper",
            "include_story_mode": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("story_summary_vi")
    assert body.get("example_portfolio_vi")
    assert body.get("biggest_drop_vi")
