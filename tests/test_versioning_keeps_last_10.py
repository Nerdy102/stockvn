from __future__ import annotations

import datetime as dt

from api_fastapi.main import create_app
from core.db.models import UserAnnotationV2, Workspace
from core.db.session import get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select


def test_versioning_keeps_last_10(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "ann_v10.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    app = create_app()
    client = TestClient(app)
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(Workspace(id="w1", user_id=None, name="Default", created_at=dt.datetime.utcnow()))
        s.commit()

    for i in range(12):
        r = client.post(
            "/chart/annotations",
            json={
                "workspace_id": "w1",
                "symbol": "AAA",
                "timeframe": "1D",
                "window_start": "2025-01-01",
                "window_end": "2025-01-31",
                "actor": "tester",
                "notes": f"save {i}",
                "shapes_json": [
                    {"type": "line", "x0": "2025-01-01", "x1": "2025-01-02", "y0": 1, "y1": 2}
                ],
            },
        )
        assert r.status_code == 200

    with Session(engine) as s:
        rows = s.exec(
            select(UserAnnotationV2)
            .where(UserAnnotationV2.workspace_id == "w1")
            .where(UserAnnotationV2.symbol == "AAA")
            .where(UserAnnotationV2.timeframe == "1D")
        ).all()
        assert len(rows) == 10
        versions = sorted([r.version for r in rows])
        assert versions[0] == 3
        assert versions[-1] == 12
