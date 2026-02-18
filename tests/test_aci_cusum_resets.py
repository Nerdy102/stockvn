from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine, select

from core.alpha_v3.conformal import (
    MODEL_ID_CP,
    _maybe_cusum_reset_bucket,
    cusum_detect_index,
    update_delayed_residuals,
)
from core.db.models import AlphaPrediction, ConformalResidual, EventLog, MlFeature, MlLabel


def test_cusum_step_shift_golden_detection_date() -> None:
    golden = json.loads(Path("tests/golden/aci_cusum_step_shift.json").read_text(encoding="utf-8"))
    idx = cusum_detect_index(
        golden["miss_series"],
        k=float(golden["cusum_k"]),
        h=float(golden["cusum_h"]),
    )
    assert idx == int(golden["expected_detection_index"])


def test_reset_logged_when_cusum_break_detected() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        start = dt.date(2024, 1, 2)
        for i in range(80):
            d = start + dt.timedelta(days=i)
            miss = 0.0 if i < 30 else 1.0
            s.add(
                ConformalResidual(
                    model_id=MODEL_ID_CP,
                    date=d,
                    symbol=f"S{i}",
                    bucket_id=0,
                    abs_residual=0.2,
                    miss=miss,
                )
            )
        s.commit()

        fired = _maybe_cusum_reset_bucket(s, bucket_id=0, matured_date=start + dt.timedelta(days=79))
        assert fired is True

        events = s.exec(
            select(EventLog)
            .where(EventLog.source == "alpha_v3_cp")
            .where(EventLog.event_type == "conformal_reset")
        ).all()
        assert len(events) == 1
        payload = events[0].payload_json or {}
        assert int(payload.get("reset_window_matured_days", 0)) == 63
        assert int(payload.get("cooldown_trading_days", 0)) == 20


def test_cooldown_blocks_repeated_resets() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        start = dt.date(2024, 1, 2)
        for i in range(90):
            d = start + dt.timedelta(days=i)
            miss = 0.0 if i < 25 else 1.0
            s.add(
                ConformalResidual(
                    model_id=MODEL_ID_CP,
                    date=d,
                    symbol=f"S{i}",
                    bucket_id=1,
                    abs_residual=0.25,
                    miss=miss,
                )
            )
        s.commit()

        d1 = start + dt.timedelta(days=80)
        fired1 = _maybe_cusum_reset_bucket(s, bucket_id=1, matured_date=d1)
        fired2 = _maybe_cusum_reset_bucket(s, bucket_id=1, matured_date=d1 + dt.timedelta(days=1))
        assert fired1 is True
        assert fired2 is False

        events = s.exec(
            select(EventLog)
            .where(EventLog.source == "alpha_v3_cp")
            .where(EventLog.event_type == "conformal_reset")
            .where(EventLog.symbol == "bucket:1")
        ).all()
        assert len(events) == 1


def test_delayed_residual_update_uses_only_matured_date() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        as_of = dt.date(2024, 5, 31)
        matured = dt.date(2024, 5, 2)

        # matured-date prediction/label pair exists
        s.add(MlFeature(symbol="AAA", as_of_date=matured, feature_version="v3", adv20_value=100.0))
        s.add(MlLabel(symbol="AAA", date=matured, y_excess=0.1, y_rank_z=0.12, label_version="v3"))
        s.add(AlphaPrediction(model_id=MODEL_ID_CP, as_of_date=matured, symbol="AAA", score=0.0, mu=0.1, uncert=0.01, pred_base=0.1))

        # future label should never be used for this as_of update
        future = as_of + dt.timedelta(days=5)
        s.add(MlFeature(symbol="AAA", as_of_date=future, feature_version="v3", adv20_value=100.0))
        s.add(MlLabel(symbol="AAA", date=future, y_excess=-0.3, y_rank_z=-0.2, label_version="v3"))
        s.add(AlphaPrediction(model_id=MODEL_ID_CP, as_of_date=future, symbol="AAA", score=0.0, mu=-0.1, uncert=0.01, pred_base=-0.1))
        s.commit()

        updated = update_delayed_residuals(s, as_of_date=as_of)
        assert updated == 1
        rows = s.exec(select(ConformalResidual).where(ConformalResidual.model_id == MODEL_ID_CP)).all()
        assert len(rows) == 1
        assert rows[0].date == matured
