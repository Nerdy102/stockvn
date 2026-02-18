from __future__ import annotations

import datetime as dt

from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import FeatureCoverage, MlFeature, MlLabel, PriceOHLCV, Ticker
from worker_scheduler.jobs import job_build_labels_v3, job_build_ml_features_v3


def _seed_prices(session: Session) -> None:
    session.add(Ticker(symbol="AAA", name="AAA", exchange="HOSE", sector="BANKING", industry="BANK"))
    session.add(Ticker(symbol="VNINDEX", name="VNINDEX", exchange="HOSE", sector="INDEX", industry="INDEX"))
    start = dt.date(2024, 1, 1)
    for i in range(80):
        day = start + dt.timedelta(days=i)
        for symbol, base in [("AAA", 10.0), ("VNINDEX", 1000.0)]:
            close = base + i * (0.1 if symbol == "AAA" else 1.0)
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
    session.commit()


def test_alpha_v3_worker_jobs_build_tables_from_fixtures() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        _seed_prices(s)
        n_labels = job_build_labels_v3(s)
        n_features = job_build_ml_features_v3(s)

        assert n_labels >= 0
        assert n_features > 0

        labels = s.exec(select(MlLabel).where(MlLabel.label_version == "v3")).all()
        features = s.exec(select(MlFeature).where(MlFeature.feature_version == "v3")).all()
        coverage = s.exec(select(FeatureCoverage).where(FeatureCoverage.feature_version == "v3")).all()
        assert len(features) > 0
        assert len(coverage) > 0
        assert len(labels) == n_labels
        assert all(l.y_rank_z == l.y_rank_z for l in labels)  # no NaN
        assert all(0.0 <= f.data_coverage_score <= 1.0 for f in features)
        assert any(f.ret_21d is not None for f in features)
