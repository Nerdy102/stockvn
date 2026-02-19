from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_confirm_execute_paper_updates_ledger() -> None:
    client = TestClient(app)
    draft = {
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
    }
    r = client.post(
        "/simple/confirm_execute",
        json={
            "portfolio_id": 99,
            "mode": "paper",
            "acknowledged_educational": True,
            "acknowledged_loss": True,
            "draft": draft,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "paper_filled"
    assert body["audit_id"].startswith("simple-audit-")
    assert Path("artifacts/audit/simple_mode_audit.jsonl").exists()
