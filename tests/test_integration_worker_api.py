from __future__ import annotations

from api_fastapi.main import create_app
from core.db.models import IndicatorValue, PriceOHLCV, Ticker
from data.etl.pipeline import ingest_from_fixtures
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select
from worker_scheduler.jobs import compute_indicators_incremental


def test_worker_ingest_compute_and_api_query() -> None:
    engine = create_engine("sqlite:///./vn_invest.db")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.merge(
            Ticker(symbol="FPT", name="FPT", exchange="HOSE", sector="Tech", industry="Software")
        )
        ingest_from_fixtures(s)
        compute_indicators_incremental(s)
        assert s.exec(select(PriceOHLCV).where(PriceOHLCV.symbol == "FPT")).first() is not None
        assert (
            s.exec(select(IndicatorValue).where(IndicatorValue.symbol == "FPT")).first() is not None
        )

    app = create_app()
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"

    p = client.get(
        "/prices",
        params={
            "symbol": "FPT",
            "timeframe": "1D",
            "start": "2025-01-01",
            "end": "2025-12-31",
            "limit": 10,
        },
    )
    assert p.status_code == 200
    assert len(p.json()) >= 1

    t = client.get("/tickers", params={"limit": 10})
    assert t.status_code == 200
    assert any(x["symbol"] == "FPT" for x in t.json())
