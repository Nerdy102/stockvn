import pandas as pd
from core.db.models import Fundamental
from data.etl.ingest import ingest_fundamentals
from sqlmodel import Session, SQLModel, create_engine, select


def test_public_date_assumption_from_period_end() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "as_of_date": "2025-03-31",
                "period_end": "2025-03-31",
                "statement_type": "quarterly",
                "sector": "Tech",
            }
        ]
    )
    with Session(engine) as s:
        ingest_fundamentals(s, df)
        row = s.exec(select(Fundamental)).first()
        assert row is not None
        assert row.public_date is not None
        assert row.public_date_is_assumed is True
