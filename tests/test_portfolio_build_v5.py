from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.alpha_v3.portfolio_v5 import (
    MAX_CLUSTER_CAP,
    MAX_SINGLE_CAP,
    RISK_BUDGET_CAPS,
    build_portfolio_v5,
    strict_project_portfolio_v5,
)


def test_portfolio_build_v5_projection_feasible() -> None:
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.01, size=(252, 6))
    clusters = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    risk_buckets = np.array(["high", "high", "mid", "mid", "low", "low"], dtype=object)

    w, cash, report = build_portfolio_v5(returns, clusters=clusters, risk_buckets=risk_buckets)

    assert np.all(w >= -1e-12)
    assert abs((float(w.sum()) + float(cash)) - 1.0) <= 1e-10
    assert max(report["violations_post"].values()) <= 1e-8
    assert report["feasible"] is True


def test_portfolio_build_v5_cluster_caps_bind_on_tiny_universe() -> None:
    base = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
    clusters = np.array(["CL1", "CL1", "CL1", "CL2"], dtype=object)
    risk_buckets = np.array(["high", "mid", "low", "low"], dtype=object)

    out = strict_project_portfolio_v5(
        base,
        clusters=clusters,
        risk_buckets=risk_buckets,
        target_sum=0.90,
    )

    assert abs(float(out.sum()) - 0.90) <= 1e-8
    assert float(out[:3].sum()) <= MAX_CLUSTER_CAP + 1e-8
    assert abs(float(out[:3].sum()) - MAX_CLUSTER_CAP) <= 1e-5
    assert np.all(out <= MAX_SINGLE_CAP + 1e-8)
    for k, cap in RISK_BUDGET_CAPS.items():
        idx = np.where(risk_buckets.astype(str) == k)[0]
        if len(idx) > 0:
            assert float(out[idx].sum()) <= cap + 1e-8


def test_portfolio_build_v5_report_keys_match_golden() -> None:
    rng = np.random.default_rng(202)
    returns = rng.normal(0.0, 0.01, size=(252, 5))
    clusters = np.array(["A", "A", "B", "B", "C"], dtype=object)
    risk_buckets = np.array(["high", "mid", "mid", "low", "low"], dtype=object)

    _w, _cash, report = build_portfolio_v5(returns, clusters=clusters, risk_buckets=risk_buckets)
    golden = json.loads(Path("tests/golden/portfolio_build_v5_report_keys.json").read_text(encoding="utf-8"))

    assert set(report.keys()) == set(golden["required_keys"])
    assert set(report["violations_pre"].keys()) == set(golden["required_violation_keys"])
    assert set(report["violations_post"].keys()) == set(golden["required_violation_keys"])
