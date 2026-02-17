from core.fees_taxes import (
    FeesTaxes,
    compute_commission,
    compute_dividend_withholding,
    compute_sell_tax,
)


def test_fee_tax_functions_and_breakdown() -> None:
    assert compute_commission(1_000_000, 0.0015) == 1500
    assert compute_sell_tax(1_000_000, 0.001) == 1000
    assert compute_dividend_withholding(200_000, 0.05) == 10_000

    ft = FeesTaxes(0.001, 0.05, 0.0015, {})
    b = ft.build_pnl_breakdown(
        gross_pnl=100_000,
        commission=1500,
        sell_tax=1000,
        dividend_withholding=500,
        slippage_cost=250,
    )
    assert b.net_pnl == 96_750
