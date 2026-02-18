from __future__ import annotations

import datetime as dt

from core.db.models import AlertV5
from sqlmodel import Session, SQLModel, create_engine, select
import worker_scheduler.jobs as jobs
from worker_scheduler.jobs import job_alert_sla_escalation_daily


def test_alert_sla_escalation_uses_trading_days(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    today = dt.date(2025, 1, 10)
    # exactly 4 trading days old (Mon -> Fri)
    alert_date = dt.date(2025, 1, 6)

    class _FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            base = dt.datetime.combine(today, dt.time(12, 0))
            return base.replace(tzinfo=tz) if tz else base

    with Session(engine) as s:
        monkeypatch.setattr(jobs.dt, "datetime", _FakeDT)
        s.add(AlertV5(symbol="AAA", date=alert_date, state="NEW", severity=1, sla_escalated=False))
        s.commit()

        out = job_alert_sla_escalation_daily(s)
        assert out["escalated"] >= 1

        row = s.exec(select(AlertV5).where(AlertV5.symbol == "AAA")).first()
        assert row is not None
        assert int(row.severity) >= 2
        assert bool(row.sla_escalated) is True
