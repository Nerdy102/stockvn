from __future__ import annotations

import numpy as np

from core.alpha_v3.models import compose_alpha_v3_score


def test_alpha_v3_score_formula_exact() -> None:
    pred_ridge = np.array([0.4, -0.2], dtype=float)
    pred_hgbr_rank = np.array([0.2, 0.5], dtype=float)
    q10 = np.array([0.1, -0.4], dtype=float)
    q50 = np.array([0.3, 0.2], dtype=float)
    q90 = np.array([0.5, 0.6], dtype=float)

    comp = compose_alpha_v3_score(pred_ridge, pred_hgbr_rank, q10, q50, q90)

    pred_base = 0.2 * pred_ridge + 0.8 * pred_hgbr_rank
    mu = q50
    uncert = np.maximum(0.0, q90 - q10)
    expected = 0.55 * pred_base + 0.45 * mu - 0.35 * uncert

    assert np.allclose(comp["pred_base"], pred_base)
    assert np.allclose(comp["mu"], mu)
    assert np.allclose(comp["uncert"], uncert)
    assert np.allclose(comp["score"], expected)
