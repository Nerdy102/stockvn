from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_audit_log_endpoint_reads_events() -> None:
    client = TestClient(app)
    payload = {
        "portfolio_id": 301,
        "user_id": "audit-user",
        "session_id": "audit-session",
        "idempotency_token": "audit-001",
        "mode": "paper",
        "acknowledged_educational": True,
        "acknowledged_loss": True,
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
            "mode": "paper",
            "off_session": False,
        },
    }
    run = client.post('/simple/confirm_execute', json=payload)
    assert run.status_code == 200

    logs = client.get('/simple/audit_logs', params={'limit': 50})
    assert logs.status_code == 200
    body = logs.json()
    assert body['count'] >= 1
    assert any(item.get('event') == 'order_state_transition' for item in body['items'])
