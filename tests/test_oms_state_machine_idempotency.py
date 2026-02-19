from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def _payload(mode: str = "paper") -> dict:
    return {
        "portfolio_id": 199,
        "user_id": "u1",
        "session_id": "s1",
        "mode": mode,
        "acknowledged_educational": True,
        "acknowledged_loss": True,
        "acknowledged_live_eligibility": mode == "live",
        "age": 30,
        "idempotency_token": "double-click-001",
        "draft": {
            "symbol": "FPT",
            "side": "BUY",
            "ui_side": "MUA",
            "qty": 100,
            "price": 10000,
            "notional": 1000000,
            "fee_tax": {"commission": 1500, "sell_tax": 0, "slippage_est": 1000, "total_cost": 2500},
            "reasons": ["test"],
            "risks": ["test"],
            "mode": mode,
            "off_session": False,
        },
    }


def test_confirm_execute_idempotent_double_click() -> None:
    client = TestClient(app)
    r1 = client.post("/simple/confirm_execute", json=_payload("paper"))
    assert r1.status_code == 200
    r2 = client.post("/simple/confirm_execute", json=_payload("paper"))
    assert r2.status_code == 200
    assert r2.json()["status"] == "idempotent_reuse"
