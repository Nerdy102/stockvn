from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

from core.monitoring.drift_monitor import DriftAlertTrade
from core.validation.readiness_report import build_readiness_report
from data.quality.models import DataQualityEvent


def test_readiness_report_hashing() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[DataQualityEvent.__table__, DriftAlertTrade.__table__])
    with Session(engine) as s:
        s.add(DataQualityEvent(market="vn", symbol="FPT", timeframe="1D", severity="warning", code="W", message="x", dataset_hash="h"))
        s.add(DriftAlertTrade(model_id="model_1", market="vn", severity="HIGH", code="DRIFT", message="x"))
        s.commit()
        w = {"stability_score": 88.0, "per_fold_metrics": [{"fold": 1}]}
        st = {"worst_case": {"net_return": -0.1, "mdd": -0.2}, "sensitivity_index": 0.3}
        r1 = build_readiness_report(walk_forward=w, stress=st, db=s, model_id="model_1", market="vn")
        r2 = build_readiness_report(walk_forward=w, stress=st, db=s, model_id="model_1", market="vn")
    assert r1["report_id"] == r2["report_id"]
