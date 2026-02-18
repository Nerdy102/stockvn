from __future__ import annotations

import datetime as dt

from core.db.models import AlertV5, DataHealthIncident
from sqlmodel import Session, SQLModel, create_engine, select
import worker_scheduler.jobs as jobs
from worker_scheduler.jobs import job_alert_sla_escalation_daily


def test_alert_incident_created_for_stale_high(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    today = dt.date(2025, 1, 10)
    # 6 trading days old to trigger incident
    alert_date = dt.date(2025, 1, 2)

    class _FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            base = dt.datetime.combine(today, dt.time(12, 0))
            return base.replace(tzinfo=tz) if tz else base

    with Session(engine) as s:
        monkeypatch.setattr(jobs.dt, "datetime", _FakeDT)
        s.add(AlertV5(symbol="BBB", date=alert_date, state="ACK", severity=3, sla_escalated=True))
        s.commit()

        out = job_alert_sla_escalation_daily(s)
        assert out["incidents"] >= 1

        inc = s.exec(select(DataHealthIncident).where(DataHealthIncident.symbol == "BBB")).first()
        assert inc is not None
        assert inc.status == "OPEN"
