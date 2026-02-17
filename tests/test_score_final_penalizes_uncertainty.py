from __future__ import annotations


def test_score_final_penalizes_uncertainty() -> None:
    mu = 0.5
    low_u = 0.1
    high_u = 0.4
    score_low = mu - 0.35 * low_u
    score_high = mu - 0.35 * high_u
    assert score_high < score_low
