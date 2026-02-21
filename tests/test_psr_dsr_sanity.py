from core.eval_lab.multiple_testing import dsr, psr


def test_psr_zero_sharpe_near_half() -> None:
    p = psr(sr_hat=0.0, sr0=0.0, n_obs=252)
    assert 0.45 <= p <= 0.55


def test_dsr_not_near_one_for_tiny_sharpe_with_many_trials() -> None:
    d = dsr(sr_hat=0.17, n_obs=249, n_trials=331)
    assert d < 0.90
