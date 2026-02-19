from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_live_risk_block_returns_correlation_and_audit() -> None:
    client = TestClient(app)
    client.post('/simple/kill_switch/toggle', json={'enabled': False, 'source': 'manual_killswitch'})

    payload = {
        "portfolio_id": 1,
        "mode": "live",
        "user_id": "risk-user",
        "session_id": "risk-session",
        "acknowledged_educational": True,
        "acknowledged_loss": True,
        "acknowledged_live_eligibility": True,
        "age": 30,
        "draft": {
            "symbol": "FPT",
            "side": "BUY",
            "ui_side": "MUA",
            "qty": 100,
            "price": 10000,
            "notional": 10**12,
            "fee_tax": {"commission": 1500, "sell_tax": 0, "slippage_est": 1000, "total_cost": 2500},
            "reasons": ["test"],
            "risks": ["test"],
            "mode": "live",
            "off_session": False,
        },
    }
    # bật live để vào nhánh risk
    import os

    os.environ["TRADING_ENV"] = "live"
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    os.environ["KILL_SWITCH"] = "false"
    os.environ["SSI_CONSUMER_ID"] = "x"
    os.environ["SSI_CONSUMER_SECRET"] = "x"
    os.environ["SSI_PRIVATE_KEY_PATH"] = "/tmp/x"
    os.environ["RISK_MAX_NOTIONAL_PER_ORDER"] = "1000"

    r = client.post('/simple/confirm_execute', json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason_code"] == "RISK_BLOCKED"
    assert str(detail.get("correlation_id", "")).startswith("ord-")

    logs = client.get('/simple/audit_logs', params={'limit': 200}).json()['items']
    matched = [x for x in logs if x.get('session') == 'risk-session' and x.get('reason_code') == 'RISK_BLOCKED']
    assert matched
    assert matched[-1].get('correlation_id') == detail.get('correlation_id')
