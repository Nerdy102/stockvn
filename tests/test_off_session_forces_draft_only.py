from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session

from core.db.session import get_database_url, get_engine
from core.monitoring.drift_monitor import DriftAlertTrade

from api_fastapi.main import app


def _clear_drift_alerts() -> None:
    engine = get_engine(get_database_url())
    SQLModel.metadata.create_all(engine, tables=[DriftAlertTrade.__table__])
    with Session(engine) as s:
        s.exec(DriftAlertTrade.__table__.delete())
        s.commit()


def test_off_session_forces_draft_only() -> None:
    _clear_drift_alerts()
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
            "off_session": True,
        },
    }
    r = client.post("/simple/confirm_execute", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["reason_code"] == "OFF_SESSION_DRAFT_ONLY"
