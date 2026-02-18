from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "packages" / "data"))
sys.path.insert(0, str(ROOT / "services" / "worker_scheduler"))

from core.db.models import DataHealthIncident
from worker_scheduler.jobs import job_bronze_ingest_replay, job_schema_monitor


def test_schema_monitor_detects_key_change(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    base = tmp_path / "lake"
    day = dt.date(2025, 1, 2)
    rows_a = [
        {
            "source": "demo",
            "channel": "trade",
            "received_ts_utc": "2025-01-02T09:00:00Z",
            "schema_version": "v1",
            "payload_json": {"symbol": "AAA", "price": 10, "qty": 1},
            "trace_json": {},
        }
    ]
    rows_b = [
        {
            "source": "demo",
            "channel": "trade",
            "received_ts_utc": "2025-01-02T10:00:00Z",
            "schema_version": "v1",
            "payload_json": {"symbol": "AAA", "price": 11, "qty": 2, "new_key": 1},
            "trace_json": {},
        }
    ]

    with Session(engine) as session:
        job_bronze_ingest_replay(session, rows_a, base_dir=str(base))
        job_schema_monitor(
            session, base_dir=str(base), dt_value=day, source="demo", channel="trade"
        )
        job_bronze_ingest_replay(session, rows_b, base_dir=str(base))
        out = job_schema_monitor(
            session, base_dir=str(base), dt_value=day, source="demo", channel="trade"
        )
        assert out["incidents"] == 1
        incident = session.exec(
            select(DataHealthIncident).order_by(DataHealthIncident.id.desc())
        ).first()
        assert incident is not None
        expected = json.loads(
            Path("tests/golden/schema_change_incident_expected.json").read_text(encoding="utf-8")
        )
        assert incident.source == expected["source"]
        assert incident.severity == expected["severity"]
        assert incident.status == expected["status"]
        assert expected["summary"] in incident.summary
