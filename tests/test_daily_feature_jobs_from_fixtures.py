from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DailyFlowFeature, DailyIntradayFeature, DailyOrderbookFeature
from data.etl.pipeline import ingest_from_fixtures
from worker_scheduler.jobs import (
    compute_daily_flow_features,
    compute_daily_orderbook_features,
    compute_intraday_daily_features,
)


def test_daily_feature_jobs_from_fixtures_smoke() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        ingest_from_fixtures(s)

        flow_up = compute_daily_flow_features(s)
        ob_up = compute_daily_orderbook_features(s)
        intr_up = compute_intraday_daily_features(s)

        assert flow_up >= 0
        assert ob_up >= 0
        assert intr_up >= 0

        _ = s.exec(select(DailyFlowFeature)).all()
        _ = s.exec(select(DailyOrderbookFeature)).all()
        _ = s.exec(select(DailyIntradayFeature)).all()
