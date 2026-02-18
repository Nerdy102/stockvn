from __future__ import annotations

import datetime as dt

from core.db.models import AlertV5, NotificationLog
from core.settings import Settings
from sqlmodel import Session, SQLModel, create_engine, select
from worker_scheduler import jobs


def test_alert_digest_respects_snooze(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    today = dt.date.today()

    with Session(engine) as s:
        s.add(AlertV5(symbol="AAA", date=today - dt.timedelta(days=1), state="NEW", severity=1))
        s.add(AlertV5(symbol="BBB", date=today - dt.timedelta(days=1), state="NEW", severity=2, snooze_until=today + dt.timedelta(days=2)))
        s.commit()

        class _FakeDT(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                base = dt.datetime.combine(today, dt.time(18, 5))
                return base.replace(tzinfo=tz) if tz else base

        monkeypatch.setattr(jobs.dt, "datetime", _FakeDT)

        settings = Settings(ALERT_EMAIL_ENABLED=True, ALERT_DIGEST_RECIPIENT="ops@example.com")
        out = jobs.job_alert_digest_daily(s, settings)
        assert out["sent"] == 1
        assert out["alerts"] == 1  # snoozed alert excluded

        log = s.exec(select(NotificationLog).where(NotificationLog.kind == "alert_digest_v5")).first()
        assert log is not None
        assert int(log.payload_json.get("count", 0)) == 1
