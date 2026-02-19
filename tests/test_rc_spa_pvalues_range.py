from __future__ import annotations

import numpy as np

from core.quant_validation_advanced.rc_spa import compute_rc_spa


def test_rc_spa_pvalues_range() -> None:
    rng = np.random.default_rng(42)
    n = 260
    bmk = rng.normal(0.0, 0.01, size=n)
    same = np.column_stack([bmk, bmk, bmk])
    rep_same, _ = compute_rc_spa(same, bmk, bootstrap_b=200, seed=42)
    assert rep_same.rc_pvalue is not None and 0.0 <= rep_same.rc_pvalue <= 1.0
    assert rep_same.spa_pvalue is not None and 0.0 <= rep_same.spa_pvalue <= 1.0
    assert rep_same.spa_pvalue >= 0.2

    better = np.column_stack([bmk + 0.005, bmk, bmk - 0.001])
    rep_better, _ = compute_rc_spa(better, bmk, bootstrap_b=200, seed=42)
    assert rep_better.spa_pvalue is not None
    assert rep_better.spa_pvalue <= 0.1
