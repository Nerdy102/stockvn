from __future__ import annotations

import datetime as dt

from api_fastapi.main import create_app
from core.db.models import EventLog
from core.db.session import create_db_and_tables, get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session


def test_uncertainty_terminal_includes_conformal_reset_events(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "cp_reset_timeline.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    create_db_and_tables(f"sqlite:///{db_path}")
    engine = get_engine(f"sqlite:///{db_path}")
    with Session(engine) as s:
        s.add(
            EventLog(
                ts_utc=dt.datetime(2025, 1, 15, 15, 0, 0),
                source="alpha_v3_cp",
                event_type="conformal_reset",
                symbol="bucket:0",
                payload_json={"before_coverage": 0.6, "after_coverage": 0.82},
                payload_hash="h1",
                run_id="run-1",
            )
        )
        s.commit()

    c = TestClient(create_app())
    r = c.get("/ml/alpha_v3_cp/uncertainty_terminal")
    assert r.status_code == 200
    body = r.json()
    ev = body.get("reset_events", [])
    assert len(ev) >= 1
    assert any(str(x.get("event_type", "")).lower() == "conformal_reset" for x in ev)
