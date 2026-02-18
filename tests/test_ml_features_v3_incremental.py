from __future__ import annotations

import datetime as dt

from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import FeatureCoverage, MlFeature, PriceOHLCV, Ticker
from worker_scheduler.jobs import job_build_ml_features_v3


def _add_price(session: Session, symbol: str, day: dt.date, close: float) -> None:
    session.add(
        PriceOHLCV(
            symbol=symbol,
            timeframe="1D",
            timestamp=dt.datetime.combine(day, dt.time(15, 0)),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1_000,
            value_vnd=close * 1_000,
            source="test",
            quality_flags={},
        )
    )


def test_ml_features_v3_incremental_only_new_dates() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(Ticker(symbol="AAA", exchange="HOSE", name="AAA", sector="BANKING", industry="BANK"))
        start = dt.date(2024, 1, 1)
        for i in range(40):
            _add_price(s, "AAA", start + dt.timedelta(days=i), 10 + i * 0.1)
        s.commit()

        first = job_build_ml_features_v3(s)
        assert first > 0
        n_features_1 = len(s.exec(select(MlFeature).where(MlFeature.feature_version == "v3")).all())
        n_cov_1 = len(s.exec(select(FeatureCoverage).where(FeatureCoverage.feature_version == "v3")).all())

        second = job_build_ml_features_v3(s)
        assert second == 0
        n_features_2 = len(s.exec(select(MlFeature).where(MlFeature.feature_version == "v3")).all())
        n_cov_2 = len(s.exec(select(FeatureCoverage).where(FeatureCoverage.feature_version == "v3")).all())
        assert n_features_2 == n_features_1
        assert n_cov_2 == n_cov_1

        _add_price(s, "AAA", start + dt.timedelta(days=40), 14.0)
        s.commit()

        third = job_build_ml_features_v3(s)
        assert third == 1
        n_features_3 = len(s.exec(select(MlFeature).where(MlFeature.feature_version == "v3")).all())
        assert n_features_3 == n_features_1 + 1
