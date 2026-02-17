from __future__ import annotations

from core.db.models import BronzeRaw, IngestState, PriceOHLCV
from data.etl.pipeline import ingest_from_fixtures
from sqlmodel import Session, SQLModel, create_engine, select


def test_ingest_from_fixtures_idempotent() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        first = ingest_from_fixtures(s)
        second = ingest_from_fixtures(s)

        bronze = s.exec(select(BronzeRaw)).all()
        silver = s.exec(select(PriceOHLCV)).all()
        state = s.exec(select(IngestState)).all()

    assert first["silver_processed"] >= 1
    assert second["silver_processed"] >= 1
    assert len(bronze) == 1
    assert len(silver) == 1
    assert len(state) == 1
