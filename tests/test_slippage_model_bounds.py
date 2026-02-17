from core.execution_model import ExecutionAssumptions, slippage_bps


def test_slippage_non_negative_and_scales() -> None:
    a = ExecutionAssumptions(base_slippage_bps=10, k1_participation=50, k2_volatility=100)
    low = slippage_bps(order_notional=1e7, adtv=1e11, atr_pct=0.01, assumptions=a)
    high = slippage_bps(order_notional=5e10, adtv=1e11, atr_pct=0.05, assumptions=a)
    assert low >= 0
    assert high > low
