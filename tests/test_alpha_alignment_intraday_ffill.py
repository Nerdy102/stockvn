from __future__ import annotations

import datetime as dt

from api_fastapi.main import create_app
from core.db.models import AlphaPrediction
from core.db.session import get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel


def test_alpha_alignment_intraday_ffill(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_ffill.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    app = create_app()
    client = TestClient(app)
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(
            AlphaPrediction(
                model_id="alpha_v3",
                as_of_date=dt.date(2025, 1, 10),
                symbol="AAA",
                score=0.1,
                mu=0.2,
                uncert=0.05,
                pred_base=0.0,
                created_at=dt.datetime.utcnow(),
            )
        )
        s.add(
            AlphaPrediction(
                model_id="alpha_v3",
                as_of_date=dt.date(2025, 1, 12),
                symbol="AAA",
                score=0.2,
                mu=0.3,
                uncert=0.05,
                pred_base=0.0,
                created_at=dt.datetime.utcnow(),
            )
        )
        s.commit()

    r = client.get(
        "/chart/alpha",
        params={
            "symbol": "AAA",
            "start": "2025-01-10",
            "end": "2025-01-13",
            "model_id": "alpha_v3",
            "timeframe": "60m",
        },
    )
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 4
    assert rows[1]["mu_norm"] is not None
