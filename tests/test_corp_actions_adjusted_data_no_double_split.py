import datetime as dt

from core.corporate_actions import CorporateActionEvent, apply_price_adjustments


def test_adjusted_data_no_double_split() -> None:
    import pandas as pd

    bars = pd.DataFrame({"date": [dt.date(2025,1,1), dt.date(2025,1,2)], "open": [100, 50], "high": [100, 50], "low": [100, 50], "close": [100, 50], "volume": [1000, 2000]})
    ev = CorporateActionEvent(symbol="AAA", action_type="SPLIT", ex_date=dt.date(2025,1,2), params={"split_factor": 2.0})
    out = apply_price_adjustments(bars, [ev], adjusted=False, total_return=False)
    assert list(out["close"]) == [100, 50]
