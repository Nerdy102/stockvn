from __future__ import annotations

import uuid
from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_paper_order_writes_oms_sequence_to_audit_log() -> None:
    client = TestClient(app)
    payload = {
        "portfolio_id": 401,
        "user_id": "oms-user",
        "session_id": "oms-session",
        "idempotency_token": f"oms-seq-{uuid.uuid4().hex[:8]}",
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
    r = client.post('/simple/confirm_execute', json=payload)
    assert r.status_code == 200

    logs = client.get('/simple/audit_logs', params={'limit': 200}).json()['items']
    by_session = [x for x in logs if x.get('session') == 'oms-session']
    pairs = {(x.get('from_state'), x.get('to_state')) for x in by_session}
    assert ('DRAFT', 'APPROVED') in pairs
    assert ('APPROVED', 'SENT') in pairs
    assert ('SENT', 'ACKED') in pairs
    assert ('ACKED', 'FILLED') in pairs
