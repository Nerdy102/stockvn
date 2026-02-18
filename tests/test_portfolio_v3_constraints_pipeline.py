from __future__ import annotations

import datetime as dt

import numpy as np
from sqlmodel import Session, SQLModel, create_engine, select

from core.alpha_v3.portfolio import (
    apply_cvar_overlay,
    construct_portfolio_v3_with_report,
    dykstra_project_weights,
    persist_constraint_report,
)
from core.db.models import RebalanceConstraintReport


def test_cvar_lp_feasible_and_fallback() -> None:
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.01, size=(252, 6))
    base = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])

    w, info = apply_cvar_overlay(returns, base, alpha=0.05, lower_mult=0.5, upper_mult=1.5)
    assert info["status"] == "optimal"
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0.5 * base - 1e-10)
    assert np.all(w <= 1.5 * base + 1e-10)

    # force bounds infeasible: sum(ub)<1
    wf, infof = apply_cvar_overlay(returns, base, alpha=0.05, lower_mult=0.5, upper_mult=0.1)
    assert infof["status"] == "fallback_bounds_infeasible"
    assert np.all(wf >= -1e-12)


def test_dykstra_projection_satisfies_all_constraints_exactly() -> None:
    base = np.array([0.40, 0.25, 0.20, 0.15])
    sectors = np.array(["A", "A", "B", "B"], dtype=object)
    ub = np.array([0.25, 0.25, 0.25, 0.20])

    out = dykstra_project_weights(
        base,
        target_sum=0.90,
        upper_bounds=ub,
        sectors=sectors,
        sector_cap=0.45,
    )

    assert np.all(out >= -1e-10)
    assert abs(float(out.sum()) - 0.90) <= 1e-8
    assert np.all(out <= ub + 1e-8)
    assert out[:2].sum() <= 0.45 + 1e-8
    assert out[2:].sum() <= 0.45 + 1e-8


def test_pipeline_deterministic_and_constraint_report_present() -> None:
    rng = np.random.default_rng(7)
    n = 35
    t = 252
    symbols = [f"S{i:02d}" for i in range(n)]
    returns = rng.normal(0.0, 0.01, size=(t, n))
    scores = np.linspace(1.0, 0.0, n)
    current_w = np.full(n, 1.0 / n)
    next_open = np.full(n, 20_000.0)
    adtv = np.linspace(2e9, 5e9, n)
    atr14 = np.full(n, 150.0)
    close = np.full(n, 20_000.0)
    spread = np.full(n, 0.0005)
    sectors = np.array(["SEC_A" if i % 2 == 0 else "SEC_B" for i in range(n)], dtype=object)

    out1 = construct_portfolio_v3_with_report(
        symbols,
        returns,
        current_w,
        1_000_000_000.0,
        next_open,
        adtv,
        atr14,
        close,
        spread,
        sectors,
        scores=scores,
        top_k=30,
    )
    out2 = construct_portfolio_v3_with_report(
        symbols,
        returns,
        current_w,
        1_000_000_000.0,
        next_open,
        adtv,
        atr14,
        close,
        spread,
        sectors,
        scores=scores,
        top_k=30,
    )

    w1, cash1, _intents1, report1 = out1
    w2, cash2, _intents2, report2 = out2

    assert np.allclose(w1, w2)
    assert np.isclose(cash1, cash2)
    assert np.all(w1 >= -1e-12)
    assert abs((w1.sum() + cash1) - 1.0) <= 1e-10
    assert report1["post_violations"]["nonneg"] <= 1e-8
    assert report1["post_violations"]["sum"] <= 1e-8
    assert report1["distance"]["l2"] >= 0.0
    assert report1["distance"]["kl"] >= 0.0
    assert report1 == report2


def test_constraint_report_stored_in_db_per_rebalance() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    report = {
        "active_constraints": ["nonneg", "sum_risky"],
        "pre_violations": {"nonneg": 0.1},
        "post_violations": {"nonneg": 0.0},
        "distance": {"l2": 0.03, "kl": 0.002},
    }

    with Session(engine) as session:
        persist_constraint_report(
            session,
            as_of_date=dt.date(2025, 1, 2),
            report=report,
            run_tag="alpha_v3_test",
        )

        rows = session.exec(select(RebalanceConstraintReport)).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.run_tag == "alpha_v3_test"
        assert row.report_json["distance"]["l2"] == 0.03
