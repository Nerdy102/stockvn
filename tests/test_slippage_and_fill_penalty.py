from core.execution_model import ExecutionAssumptions, execution_fill_ratio, slippage_bps


def test_slippage_and_fill_penalty() -> None:
    s = ExecutionAssumptions(base_slippage_bps=10.0)
    bps = slippage_bps(order_notional=1e9, adtv=5e9, atr_pct=0.02, assumptions=s)
    assert bps > 10.0
    assert execution_fill_ratio('BUY', at_upper_limit=True, at_lower_limit=False, assumptions=s) == 0.2
    assert execution_fill_ratio('SELL', at_upper_limit=False, at_lower_limit=True, assumptions=s) == 0.2
