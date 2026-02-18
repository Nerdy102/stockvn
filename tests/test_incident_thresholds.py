from __future__ import annotations

import json
import uuid
from pathlib import Path

from core.db.models import DataHealthIncident
from core.db.session import create_db_and_tables, get_engine
from sqlmodel import Session, select
from worker_scheduler.jobs import job_realtime_incident_monitor


def test_incident_thresholds_create_deterministic_incidents() -> None:
    base = Path("artifacts") / f"test_incident_thresholds_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)

    gateway = {
        "service": "gateway",
        "window_s": 300,
        "ingest_lag_s_p95": 6.2,
        "bar_build_latency_s_p95": 0.1,
        "signal_latency_s_p95": 0.1,
        "redis_stream_pending": 1000,
    }
    bars = {
        "service": "bar_builder",
        "window_s": 300,
        "ingest_lag_s_p95": 0.5,
        "bar_build_latency_s_p95": 3.5,
        "signal_latency_s_p95": 0.2,
        "redis_stream_pending": 60000,
    }
    signal = {
        "service": "signal_engine",
        "window_s": 300,
        "ingest_lag_s_p95": 0.5,
        "bar_build_latency_s_p95": 0.2,
        "signal_latency_s_p95": 5.5,
        "redis_stream_pending": 10,
    }

    gp = base / "gateway.json"
    bp = base / "bars.json"
    sp = base / "signal.json"
    gp.write_text(json.dumps(gateway), encoding="utf-8")
    bp.write_text(json.dumps(bars), encoding="utf-8")
    sp.write_text(json.dumps(signal), encoding="utf-8")

    db_url = f"sqlite:///./artifacts/test_incident_thresholds_{uuid.uuid4().hex}.db"
    create_db_and_tables(db_url)
    engine = get_engine(db_url)

    with Session(engine) as session:
        out = job_realtime_incident_monitor(
            session,
            snapshot_paths=[str(gp), str(bp), str(sp)],
        )
        assert out["incidents_created"] == 4
        rows = session.exec(select(DataHealthIncident).order_by(DataHealthIncident.id.asc())).all()
        assert len(rows) == 4
        runbooks = {r.runbook_section for r in rows}
        assert "runbook:realtime_lag" in runbooks
        assert "runbook:bar_perf" in runbooks
        assert "runbook:signal_perf" in runbooks
        assert "runbook:redis_backlog" in runbooks
