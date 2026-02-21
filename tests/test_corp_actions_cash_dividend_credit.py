import datetime as dt

from core.corporate_actions import CorporateActionEvent, apply_ca_to_position


def test_cash_dividend_credit() -> None:
    ev = CorporateActionEvent(symbol="AAA", action_type="CASH_DIVIDEND", ex_date=dt.date(2025, 1, 2), params={"cash_per_share": 1000})
    out = apply_ca_to_position(qty_before=10, avg_cost_before=20_000, ex_date_price=21_000, action=ev)
    assert out["cash_delta"] > 0
