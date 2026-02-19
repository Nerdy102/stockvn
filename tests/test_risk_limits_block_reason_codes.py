from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_risk_limits_block_reason_codes(monkeypatch) -> None:
    monkeypatch.setenv("RISK_MAX_NOTIONAL_PER_ORDER", "1000")
    client = TestClient(app)
    payload = {
        "portfolio_id": 1,
        "mode": "paper",
        "acknowledged_educational": True,
        "acknowledged_loss": True,
        "acknowledged_live_eligibility": False,
        "age": 30,
        "draft": {
            "symbol": "FPT",
            "side": "BUY",
            "ui_side": "MUA",
            "qty": 100,
            "price": 10000,
            "notional": 1000000,
            "fee_tax": {
                "commission": 1500,
                "sell_tax": 0,
                "slippage_est": 1000,
                "total_cost": 2500,
            },
            "reasons": ["test"],
            "risks": ["test"],
            "mode": "paper",
            "off_session": False,
        },
    }
    r = client.post("/simple/confirm_execute", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason_code"] == "RISK_BLOCKED"
    assert "RISK_MAX_NOTIONAL_EXCEEDED" in detail["reasons"]
