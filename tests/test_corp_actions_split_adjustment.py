import datetime as dt

from core.corporate_actions import CorporateActionEvent, apply_ca_to_position


def test_split_adjustment() -> None:
    ev = CorporateActionEvent(symbol="AAA", action_type="SPLIT", ex_date=dt.date(2025, 1, 2), params={"split_factor": 2.0})
    out = apply_ca_to_position(qty_before=100, avg_cost_before=20_000, ex_date_price=21_000, action=ev)
    assert out["qty_after"] == 200
    assert out["avg_cost_after"] == 10_000
