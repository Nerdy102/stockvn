from __future__ import annotations

from pathlib import Path

from api_fastapi.main import create_app
from data.etl.pipeline import ingest_from_fixtures
from fastapi.testclient import TestClient
from sqlmodel import Session

from core.db.session import create_db_and_tables, get_engine
from worker_scheduler.jobs import compute_indicators_incremental


def test_integration_ingest_compute_train_backtest_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_url = f"sqlite:///{tmp_path / 'integration_ml_pipeline.db'}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    create_db_and_tables(db_url)
    engine = get_engine(db_url)
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
