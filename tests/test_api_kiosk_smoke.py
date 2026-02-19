from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_simple_kiosk_payload_smoke() -> None:
    client = TestClient(app)
    r = client.get("/simple/kiosk", params={"universe": "VN30", "limit_signals": 10})
    assert r.status_code == 200
    body = r.json()
    assert "as_of_date" in body
    assert isinstance(body.get("market_today_text"), list)
    assert isinstance(body.get("buy_candidates"), list)
    assert isinstance(body.get("sell_candidates"), list)
    assert len(body.get("buy_candidates", [])) <= 10
    assert len(body.get("sell_candidates", [])) <= 10
    assert len(body.get("model_cards", [])) == 3
    assert "paper_summary" in body
    assert isinstance(body.get("disclaimers"), list)
