from __future__ import annotations

from sqlmodel import Session, select

from core.db.models import DiagnosticsMetric, DiagnosticsRun
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

        latest_run = s.exec(select(DiagnosticsRun).order_by(DiagnosticsRun.id.desc())).first()
        assert latest_run is not None
        metrics = s.exec(select(DiagnosticsMetric).where(DiagnosticsMetric.run_id == latest_run.run_id)).all()
        assert len(metrics) > 0
        names = {m.metric_name for m in metrics}
        assert "rank_ic_mean" in names
