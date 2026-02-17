from __future__ import annotations

from api_fastapi.main import create_app
from data.etl.pipeline import ingest_from_fixtures
from fastapi.testclient import TestClient
from sqlmodel import Session

from core.db.session import get_engine
from worker_scheduler.jobs import compute_indicators_incremental


def test_integration_ingest_compute_train_backtest_smoke() -> None:
    engine = get_engine("sqlite:///./vn_invest.db")
    with Session(engine) as s:
        ingest_from_fixtures(s)
        compute_indicators_incremental(s)

    c = TestClient(create_app())
    assert c.get("/health").status_code == 200
    assert "ensemble_v2" in c.get("/ml/models").json()
    tr = c.post("/ml/train")
    assert tr.status_code == 200
    bt = c.post("/ml/backtest", json={"mode": "smoke"})
    assert bt.status_code == 200
    body = bt.json()
    assert "walk_forward" in body and "stress" in body and "sensitivity" in body
