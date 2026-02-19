from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_risk_daily_loss_blocks_live(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("TRADING_ENV", "live")
    monkeypatch.setenv("KILL_SWITCH", "false")
    monkeypatch.setenv("SSI_CONSUMER_ID", "x")
    monkeypatch.setenv("SSI_CONSUMER_SECRET", "x")
    monkeypatch.setenv("SSI_PRIVATE_KEY_PATH", "/tmp/x")
    monkeypatch.setenv("RISK_REALIZED_DAILY_PNL", "-30000000")
    monkeypatch.setenv("RISK_NAV", "1000000000")
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS_PCT", "0.02")
    client = TestClient(app)
    client.post("/simple/kill_switch/toggle", json={"enabled": False, "source": "manual_killswitch"})
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
    r = client.post("/simple/confirm_execute", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["reason_code"] == "RISK_BLOCKED"
    assert "RISK_MAX_DAILY_LOSS_EXCEEDED" in r.json()["detail"]["reasons"]
