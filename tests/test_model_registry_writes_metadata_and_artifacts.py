from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import AlphaModel, MlFeature, MlLabel
from worker_scheduler.jobs import train_alpha_v3


def _seed_training_data(session: Session, n: int = 420) -> None:
    rng = np.random.default_rng(123)
    start = dt.date(2025, 1, 1)
    for i in range(n):
        day = start + dt.timedelta(days=i // 6)
        symbol = f"SYM{i % 6}"
        feats = {
            "ret_1d": float(rng.normal()),
            "ret_5d": float(rng.normal()),
            "vol_20d": float(abs(rng.normal())),
            "rsi14": float(rng.uniform(0, 100)),
            "ema50_slope": float(rng.normal()),
        }
        y = 0.3 * feats["ret_1d"] - 0.15 * feats["ret_5d"] + 0.05 * feats["ema50_slope"]
        session.add(MlFeature(symbol=symbol, as_of_date=day, feature_version="v3", features_json=feats))
        session.add(MlLabel(symbol=symbol, date=day, y_excess=y, y_rank_z=y, label_version="v3"))
    session.commit()


def test_model_registry_writes_metadata_and_artifacts(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        _seed_training_data(session)
        res = train_alpha_v3(session, artifact_root=str(tmp_path / "artifacts" / "models" / "alpha_v3"))
        assert res is not None

        model_row = session.exec(select(AlphaModel).where(AlphaModel.model_id == "alpha_v3")).first()
        assert model_row is not None
        model_dir = Path(model_row.artifact_path)
        assert model_dir.exists()
        for fname in ["ridge.joblib", "hgbr.joblib", "q10.joblib", "q50.joblib", "q90.joblib", "metadata.json"]:
            assert (model_dir / fname).exists()

        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["feature_version"] == "v3"
        assert metadata["label_version"] == "v3"
        assert metadata["config_hash"] == model_row.config_hash
        assert metadata["train_start"] == str(model_row.train_start)
        assert metadata["train_end"] == str(model_row.train_end)
        assert isinstance(metadata.get("feature_columns"), list)
        assert len(metadata["feature_columns"]) > 0
