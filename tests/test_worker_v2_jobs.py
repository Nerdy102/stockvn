from __future__ import annotations

from sqlmodel import Session

from core.db.session import get_engine
from data.etl.pipeline import ingest_from_fixtures
from worker_scheduler.jobs import run_diagnostics_v2, train_models_v2


def test_worker_v2_train_and_diagnostics_smoke() -> None:
    engine = get_engine('sqlite:///./vn_invest.db')
    with Session(engine) as s:
        ingest_from_fixtures(s)
        trained = train_models_v2(s)
        assert trained >= 0
        diag = run_diagnostics_v2(s)
        assert diag >= 0
