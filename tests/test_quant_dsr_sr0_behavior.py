from __future__ import annotations

from core.quant_stats.psr_dsr import deflated_sharpe_ratio


def test_quant_dsr_sr0_behavior() -> None:
    sr_trials = [0.1, 0.2, 0.0, 0.15, 0.3]
    dsr_small, sr0_small, _ = deflated_sharpe_ratio(0.4, 252, 0.0, 3.0, sr_trials, n_eff=5)
    dsr_big, sr0_big, _ = deflated_sharpe_ratio(0.4, 252, 0.0, 3.0, sr_trials, n_eff=50)
    assert sr0_big >= sr0_small
    assert dsr_big <= dsr_small + 1e-12
