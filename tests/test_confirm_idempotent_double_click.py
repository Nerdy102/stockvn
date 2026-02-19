from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def _payload() -> dict:
    return {
        "portfolio_id": 301,
        "user_id": "double-click-user",
        "session_id": "double-click-session",
        "mode": "paper",
        "acknowledged_educational": True,
        "acknowledged_loss": True,
        "acknowledged_live_eligibility": False,
        "age": 30,
        "idempotency_token": "same-token-double-click",
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


def test_confirm_idempotent_double_click() -> None:
    client = TestClient(app)
    first = client.post("/simple/confirm_execute", json=_payload())
    assert first.status_code == 200
    second = client.post("/simple/confirm_execute", json=_payload())
    assert second.status_code == 200
    assert second.json()["status"] == "idempotent_reuse"
