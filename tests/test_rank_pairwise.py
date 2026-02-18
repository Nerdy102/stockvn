from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import AlphaPrediction, MlFeature, MlLabel
from core.ml.rank_pairwise import (
    PairwiseRankConfig,
    predict_rank_score,
    purged_kfold_embargo_date_splits,
    sample_pairs_for_date,
    train_pairwise_ranker,
    validate_purged_cv_no_leakage,
)
from worker_scheduler.jobs import job_train_alpha_rankpair_v1


def test_pair_sampling_is_deterministic() -> None:
    x = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, -0.2]], dtype=float)
    y = np.array([0.1, 0.3, -0.4], dtype=float)
    day = dt.date(2024, 1, 10)

    xa, ya = sample_pairs_for_date(x, y, day, m_pairs=20)
    xb, yb = sample_pairs_for_date(x, y, day, m_pairs=20)

    assert np.array_equal(xa, xb)
    assert np.array_equal(ya, yb)


def test_purged_cv_splits_no_leakage() -> None:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(80)]
    splits = purged_kfold_embargo_date_splits(
        dates=dates,
        n_splits=5,
        purge_horizon_days=21,
        embargo_days=5,
    )
    assert len(splits) == 5
    assert validate_purged_cv_no_leakage(splits, purge_horizon_days=21, embargo_days=5)


def test_pairwise_monotonic_toy() -> None:
    rows = []
    start = dt.date(2024, 1, 1)
    for i in range(40):
        d = start + dt.timedelta(days=i)
        rows.append({"symbol": "A", "as_of_date": d, "f1": 2.0, "f2": 1.0, "y_rank_z": 1.0})
        rows.append({"symbol": "B", "as_of_date": d, "f1": 1.0, "f2": 0.0, "y_rank_z": -1.0})

    frame = pd.DataFrame(rows)
    model = train_pairwise_ranker(frame, feature_columns=["f1", "f2"], config=PairwiseRankConfig(pairs_per_date=100))
    assert model is not None

    test = pd.DataFrame(
        [
            {"symbol": "A", "as_of_date": dt.date(2024, 3, 1), "f1": 2.0, "f2": 1.0},
            {"symbol": "B", "as_of_date": dt.date(2024, 3, 1), "f1": 1.0, "f2": 0.0},
        ]
    )
    pred = predict_rank_score(model, test)
    score_a = float(pred.loc[pred["symbol"] == "A", "score_z"].iloc[0])
    score_b = float(pred.loc[pred["symbol"] == "B", "score_z"].iloc[0])
    assert score_a > score_b


def test_rankpair_integration_smoke_train_predict() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        start = dt.date(2024, 1, 1)
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        for i in range(120):
            d = start + dt.timedelta(days=i)
            for j, sym in enumerate(symbols):
                ret = float((j + 1) * 0.01 + 0.0001 * i)
                session.add(
                    MlFeature(
                        symbol=sym,
                        as_of_date=d,
                        feature_version="v3",
                        ret_1d=ret,
                        ret_5d=ret * 2,
                        vol_20d=1.0 / (j + 1),
                        rsi14=40.0 + j,
                        ema50_slope=ret,
                    )
                )
                session.add(
                    MlLabel(
                        symbol=sym,
                        date=d,
                        y_excess=ret,
                        y_rank_z=float(j - 1.5),
                        label_version="v3",
                    )
                )
        session.commit()

        res = job_train_alpha_rankpair_v1(session)
        assert res["trained"] == 1
        assert res["predictions"] > 0

        rows = session.exec(select(AlphaPrediction).where(AlphaPrediction.model_id == "alpha_rankpair_v1")).all()
        assert len(rows) > 0
        assert all(np.isfinite(r.score) for r in rows)
        assert all(r.score == r.score for r in rows)
