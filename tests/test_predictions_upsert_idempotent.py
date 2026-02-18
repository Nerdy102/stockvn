from __future__ import annotations

import datetime as dt

import numpy as np
from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import AlphaPrediction, MlFeature, MlLabel
from worker_scheduler.jobs import predict_alpha_v3, train_alpha_v3


def _seed_training_and_prediction_data(session: Session, n: int = 420) -> dt.date:
    rng = np.random.default_rng(7)
    start = dt.date(2025, 2, 1)
    latest = start
    for i in range(n):
        day = start + dt.timedelta(days=i // 7)
        latest = max(latest, day)
        symbol = f"SYM{i % 7}"
        feats = {
            "ret_1d": float(rng.normal()),
            "ret_5d": float(rng.normal()),
            "vol_20d": float(abs(rng.normal())),
            "rsi14": float(rng.uniform(0, 100)),
            "ema50_slope": float(rng.normal()),
        }
        y = 0.25 * feats["ret_1d"] - 0.2 * feats["ret_5d"]
        session.add(MlFeature(symbol=symbol, as_of_date=day, feature_version="v3", **feats))
        session.add(MlLabel(symbol=symbol, date=day, y_excess=y, y_rank_z=y, label_version="v3"))
    session.commit()
    return latest


def test_predictions_upsert_idempotent(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        pred_date = _seed_training_and_prediction_data(session)
        trained = train_alpha_v3(session, artifact_root=str(tmp_path / "artifacts" / "models" / "alpha_v3"))
        assert trained is not None

        n1 = predict_alpha_v3(session, as_of_date=pred_date, version=trained["version"])
        n2 = predict_alpha_v3(session, as_of_date=pred_date, version=trained["version"])
        rows = session.exec(
            select(AlphaPrediction)
            .where(AlphaPrediction.model_id == "alpha_v3")
            .where(AlphaPrediction.as_of_date == pred_date)
        ).all()
        first_created = {r.symbol: r.created_at for r in rows}

        _ = predict_alpha_v3(session, as_of_date=pred_date, version=trained["version"])
        rows_after = session.exec(
            select(AlphaPrediction)
            .where(AlphaPrediction.model_id == "alpha_v3")
            .where(AlphaPrediction.as_of_date == pred_date)
        ).all()

        assert n1 == n2
        assert len(rows) == n1
        assert all(first_created[r.symbol] == r.created_at for r in rows_after)


def test_predictions_upsert_idempotent_handles_feature_drift(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        pred_date = _seed_training_and_prediction_data(session)
        trained = train_alpha_v3(session, artifact_root=str(tmp_path / "artifacts" / "models" / "alpha_v3"))
        assert trained is not None

        # add live row with missing and extra columns to ensure predict aligns to training columns
        session.add(
            MlFeature(
                symbol="DRIFT",
                as_of_date=pred_date,
                feature_version="v3",
                ret_1d=0.1,
                ret_5d=-0.2,
            )
        )
        session.commit()

        n = predict_alpha_v3(session, as_of_date=pred_date, version=trained["version"])
        rows = session.exec(
            select(AlphaPrediction)
            .where(AlphaPrediction.model_id == "alpha_v3")
            .where(AlphaPrediction.as_of_date == pred_date)
        ).all()
        assert n == len(rows)
        assert any(r.symbol == "DRIFT" for r in rows)
