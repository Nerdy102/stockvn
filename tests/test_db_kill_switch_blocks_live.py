from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_db_kill_switch_blocks_live(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("TRADING_ENV", "live")
    client = TestClient(app)
    t = client.post('/simple/kill_switch/toggle', json={'enabled': True, 'source': 'manual_killswitch'})
    assert t.status_code == 200

    payload = {
        "portfolio_id": 1,
        "mode": "live",
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
            "notional": 1000000,
            "fee_tax": {"commission": 1500, "sell_tax": 0, "slippage_est": 1000, "total_cost": 2500},
            "reasons": ["test"],
            "risks": ["test"],
            "mode": "live",
            "off_session": False,
        },
    }
    r = client.post('/simple/confirm_execute', json=payload)
    assert r.status_code == 422
    assert r.json()['detail']['reason_code'] == 'LIVE_BLOCKED_TRADING_ENV' or r.json()['detail']['reason_code'] == 'LIVE_BLOCKED_KILL_SWITCH_RUNTIME'
