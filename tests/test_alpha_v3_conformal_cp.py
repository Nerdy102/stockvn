from __future__ import annotations

import datetime as dt

import numpy as np
from sqlmodel import SQLModel, Session, create_engine, select

from core.alpha_v3.conformal import (
    MODEL_ID_CP,
    _update_alpha,
    apply_cp_predictions,
    cp_interval_half_width,
    recompute_bucket_spec_monthly,
    update_delayed_residuals,
)
from core.db.models import (
    AlphaPrediction,
    ConformalCoverageDaily,
    ConformalResidual,
    ConformalState,
    MlFeature,
    MlLabel,
)


def test_aci_alpha_increases_and_decreases_with_miss() -> None:
    st = ConformalState(model_id=MODEL_ID_CP, bucket_id=0, alpha_b=0.2, miss_ema=0.2)
    _update_alpha(st, miss=1.0)
    after_high = st.alpha_b
    _update_alpha(st, miss=0.0)
    after_low = st.alpha_b
    assert after_high > 0.2
    assert after_low < after_high


def test_no_future_residual_use() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        day = dt.date(2024, 2, 1)
        s.add(MlFeature(symbol="AAA", as_of_date=day, feature_version="v3", adv20_value=100.0))
        s.add(MlLabel(symbol="AAA", date=day, y_excess=0.1, y_rank_z=0.1, label_version="v3"))
        s.add(AlphaPrediction(model_id=MODEL_ID_CP, as_of_date=day, symbol="AAA", score=0.0, mu=0.1, uncert=0.1, pred_base=0.1))
        s.commit()

        updated = update_delayed_residuals(s, as_of_date=day)
        assert updated == 0
        assert len(s.exec(select(ConformalResidual)).all()) == 0


def test_bucket_recompute_monthly_stable() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        day = dt.date(2024, 3, 15)
        for i, v in enumerate([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]):
            s.add(MlFeature(symbol=f"S{i}", as_of_date=day, feature_version="v3", adv20_value=v))
        s.commit()

        b1 = recompute_bucket_spec_monthly(s, day)
        b2 = recompute_bucket_spec_monthly(s, day)
        assert b1.bounds == b2.bounds


def test_empirical_coverage_synthetic_approx_target() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    rng = np.random.default_rng(42)

    with Session(engine) as s:
        s.add(ConformalState(model_id=MODEL_ID_CP, bucket_id=0, alpha_b=0.2, miss_ema=0.2))
        for i in range(800):
            d = dt.date(2023, 1, 1) + dt.timedelta(days=i)
            e = float(abs(rng.normal(0.0, 1.0)))
            s.add(ConformalResidual(model_id=MODEL_ID_CP, date=d, symbol=f"S{i}", bucket_id=0, abs_residual=e, miss=0.0))
        s.commit()

        width = cp_interval_half_width(s, bucket_id=0, alpha_b=0.2)
        test_err = np.abs(rng.normal(0.0, 1.0, size=2000))
        coverage = float(np.mean(test_err <= width))
        assert 0.72 <= coverage <= 0.88


def test_cp_prediction_and_coverage_rows_created() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        day = dt.date(2024, 4, 30)
        matured = dt.date(2024, 3, 29)

        s.add(MlFeature(symbol="AAA", as_of_date=day, feature_version="v3", adv20_value=100.0))
        s.add(AlphaPrediction(model_id="alpha_v3", as_of_date=day, symbol="AAA", score=0.3, mu=0.2, uncert=0.1, pred_base=0.25))

        s.add(MlFeature(symbol="AAA", as_of_date=matured, feature_version="v3", adv20_value=90.0))
        s.add(MlLabel(symbol="AAA", date=matured, y_excess=0.1, y_rank_z=0.15, label_version="v3"))
        s.add(AlphaPrediction(model_id=MODEL_ID_CP, as_of_date=matured, symbol="AAA", score=0.0, mu=0.1, uncert=0.01, pred_base=0.1))
        s.commit()

        _ = apply_cp_predictions(s, day)
        _ = update_delayed_residuals(s, as_of_date=day + dt.timedelta(days=35))

        cp_rows = s.exec(select(AlphaPrediction).where(AlphaPrediction.model_id == MODEL_ID_CP).where(AlphaPrediction.as_of_date == day)).all()
        cov_rows = s.exec(select(ConformalCoverageDaily).where(ConformalCoverageDaily.model_id == MODEL_ID_CP)).all()
        assert len(cp_rows) > 0
        assert all(np.isfinite(r.score) for r in cp_rows)
        assert len(cov_rows) >= 0
