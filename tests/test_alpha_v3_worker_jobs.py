from __future__ import annotations

from sqlmodel import SQLModel, Session, create_engine, select

from core.db.models import MlFeature, MlLabel
from data.etl.pipeline import ingest_from_fixtures
from worker_scheduler.jobs import job_build_labels_v3, job_build_ml_features_v3


def test_alpha_v3_worker_jobs_build_tables_from_fixtures() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        ingest_from_fixtures(s)
        n_labels = job_build_labels_v3(s)
        n_features = job_build_ml_features_v3(s)

        assert n_labels >= 0
        assert n_features > 0

        labels = s.exec(select(MlLabel).where(MlLabel.label_version == "v3")).all()
        features = s.exec(select(MlFeature).where(MlFeature.feature_version == "v3")).all()
        assert len(features) > 0
        assert len(labels) == n_labels
        assert all(l.y_rank_z == l.y_rank_z for l in labels)  # no NaN
        assert all(isinstance(f.features_json, dict) for f in features)
        assert "close" not in features[0].features_json
        assert "ret_21d" in features[0].features_json
