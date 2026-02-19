from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel

from api_fastapi.main import app
from core.db.session import get_database_url, get_engine
from core.monitoring.drift_monitor import DriftAlertTrade


def test_execute_blocked_on_high_drift() -> None:
    engine = get_engine(get_database_url())
    SQLModel.metadata.create_all(engine, tables=[DriftAlertTrade.__table__])
    with Session(engine) as s:
        s.add(DriftAlertTrade(model_id="model_1", market="vn", severity="HIGH", code="DRIFT_SLIPPAGE_RATIO_HIGH", message="x"))
        s.commit()

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
            "fee_tax": {"commission": 1500, "sell_tax": 0, "slippage_est": 1000, "total_cost": 2500},
            "reasons": ["test"],
            "risks": ["test"],
            "mode": "paper",
            "off_session": False,
        },
    }
    r = client.post("/simple/confirm_execute", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["reason_code"] == "DRIFT_HIGH_BLOCKED"
