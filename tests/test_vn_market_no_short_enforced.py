from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_vn_market_no_short_enforced() -> None:
    client = TestClient(app)
    r = client.post(
        "/simple/run_signal",
        json={
            "symbol": "FPT",
            "timeframe": "1D",
            "model_id": "model_2",
            "market": "vn",
            "trading_type": "perp_paper",
        },
    )
    assert r.status_code == 200
    draft = r.json().get("draft")
    if draft is not None:
        assert draft["side"] != "SHORT"
