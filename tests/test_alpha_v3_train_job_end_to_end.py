from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import AlphaModel, AlphaPrediction, MlFeature, MlLabel
from worker_scheduler import jobs


def _seed_data(session: Session, n: int = 420) -> None:
    rng = np.random.default_rng(101)
    start = dt.date(2025, 3, 1)
    for i in range(n):
        day = start + dt.timedelta(days=i // 5)
        symbol = f"SYM{i % 5}"
        feats = {
            "ret_1d": float(rng.normal()),
            "ret_5d": float(rng.normal()),
            "vol_20d": float(abs(rng.normal())),
            "rsi14": float(rng.uniform(0, 100)),
            "ema50_slope": float(rng.normal()),
        }
        y = 0.2 * feats["ret_1d"] + 0.1 * feats["ema50_slope"]
        session.add(MlFeature(symbol=symbol, as_of_date=day, feature_version="v3", **feats))
        session.add(MlLabel(symbol=symbol, date=day, y_excess=y, y_rank_z=y, label_version="v3"))
    session.commit()


def test_alpha_v3_train_job_end_to_end(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        _seed_data(session)

        original_train = jobs.train_alpha_v3

        def _patched_train(s, artifact_root="artifacts/models/alpha_v3"):
            return original_train(s, artifact_root=str(tmp_path / "artifacts" / "models" / "alpha_v3"))

        jobs.train_alpha_v3 = _patched_train
        try:
            result = jobs.job_train_alpha_v3(session)
        finally:
            jobs.train_alpha_v3 = original_train

        assert result["trained"] == 1
        assert result["predictions"] > 0

        mdl = session.exec(select(AlphaModel).where(AlphaModel.model_id == "alpha_v3")).first()
        assert mdl is not None
        assert Path(mdl.artifact_path).exists()

        preds = session.exec(select(AlphaPrediction).where(AlphaPrediction.model_id == "alpha_v3")).all()
        assert len(preds) == result["predictions"]
