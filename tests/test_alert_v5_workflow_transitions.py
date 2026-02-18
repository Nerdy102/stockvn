from __future__ import annotations

import datetime as dt
import uuid

from api_fastapi.main import create_app
from core.db.models import AlertV5
from core.db.session import get_engine
from core.settings import get_settings
from fastapi.testclient import TestClient
from sqlmodel import Session, select


def test_alert_v5_workflow_transitions() -> None:
    engine = get_engine(get_settings().DATABASE_URL)
    sym = f"ZZZ_{uuid.uuid4().hex[:8]}"
    with Session(engine) as s:
        s.add(AlertV5(symbol=sym, date=dt.date(2025, 1, 10), state="NEW", severity=1))
        s.commit()
        row = s.exec(select(AlertV5).where(AlertV5.symbol == sym).order_by(AlertV5.id.desc())).first()
        alert_id = int(row.id)

    c = TestClient(create_app())
    r1 = c.post(f"/alerts/v5/{alert_id}/action", json={"action": "ACK"})
    assert r1.status_code == 200
    r2 = c.post(f"/alerts/v5/{alert_id}/action", json={"action": "RESOLVE"})
    assert r2.status_code == 200

    r_bad = c.post(f"/alerts/v5/{alert_id}/action", json={"action": "ACK"})
    assert r_bad.status_code == 400
