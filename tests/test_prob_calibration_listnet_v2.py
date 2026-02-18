from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from sqlmodel import SQLModel, Session, create_engine, select

from core.alpha_v3.calibration import compute_probability_calibration_metrics
from core.db.models import AlphaPrediction, DataHealthIncident, MlLabel
from core.ml.prob_calibration import fit_calibrated_probability_model
from worker_scheduler.jobs import job_prob_calibration_governance_listnet_v2


def test_prob_calibration_hand_example_matches_golden() -> None:
    golden = json.loads(Path("tests/golden/prob_calibration_hand_example.json").read_text(encoding="utf-8"))
    out = compute_probability_calibration_metrics(golden["probs"], golden["outcomes"], bins=int(golden["bins"]))
    assert abs(float(out["brier"]) - float(golden["expected"]["brier"])) <= 1e-12
    assert abs(float(out["ece"]) - float(golden["expected"]["ece"])) <= 1e-12


def test_brier_improves_after_calibration_on_synthetic() -> None:
    rng = np.random.default_rng(42)
    s_train = rng.normal(0.0, 1.0, size=500)
    p_true_train = 1.0 / (1.0 + np.exp(-1.5 * s_train))
    z_train = rng.binomial(1, p_true_train)

    s_val = rng.normal(0.0, 1.0, size=300)
    p_true_val = 1.0 / (1.0 + np.exp(-1.5 * s_val))
    z_val = rng.binomial(1, p_true_val)

    model = fit_calibrated_probability_model(s_train, z_train, s_val, z_val)
    p_raw = model.p_raw(s_val)
    p_cal = model.p_cal(s_val)

    brier_raw = float(np.mean((p_raw - z_val) ** 2))
    brier_cal = float(np.mean((p_cal - z_val) ** 2))
    assert brier_cal <= brier_raw + 1e-8


def test_no_leakage_isotonic_uses_only_validation() -> None:
    rng = np.random.default_rng(7)
    s_train = rng.normal(size=120)
    z_train = rng.binomial(1, 1.0 / (1.0 + np.exp(-s_train)))

    s_val = np.linspace(-2, 2, 60)
    z_val_a = np.zeros_like(s_val)
    z_val_b = np.ones_like(s_val)

    m_a = fit_calibrated_probability_model(s_train, z_train, s_val, z_val_a)
    m_b = fit_calibrated_probability_model(s_train, z_train, s_val, z_val_b)

    assert abs(m_a.a - m_b.a) < 1e-12
    assert abs(m_a.b - m_b.b) < 1e-12
    assert not np.allclose(m_a.iso_y, m_b.iso_y)


def test_governance_event_triggers_when_rolling_ece_above_threshold() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        start = dt.date(2025, 1, 1)
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        for i in range(25):
            d = start + dt.timedelta(days=i)
            for j, sym in enumerate(symbols):
                # highly miscalibrated: p_cal high but outcomes mostly 0
                p_cal = 0.9 if j < 4 else 0.8
                y_excess = -0.01 if j < 4 else -0.02
                session.add(
                    AlphaPrediction(
                        model_id="alpha_listnet_v2",
                        as_of_date=d,
                        symbol=sym,
                        score=0.0,
                        mu=p_cal,
                        uncert=0.95,
                        pred_base=0.0,
                    )
                )
                session.add(
                    MlLabel(
                        symbol=sym,
                        date=d,
                        y_excess=y_excess,
                        y_rank_z=float(j),
                        label_version="v3",
                    )
                )
        session.commit()

        out = job_prob_calibration_governance_listnet_v2(session, lookback_days=20, ece_threshold=0.05)
        assert bool(out["triggered"]) is True

        inc = session.exec(select(DataHealthIncident).where(DataHealthIncident.source == "prob_calibration")).first()
        assert inc is not None
        assert inc.status == "OPEN"
