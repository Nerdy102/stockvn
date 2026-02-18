import datetime as dt

import pytest
import pandas as pd

from core.corporate_actions import (
    WITHHOLDING_TAX_RATE,
    CorporateActionEvent,
    adjust_prices,
    apply_ca_to_position,
)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3), dt.date(2025, 1, 4)],
            "open": [100.0, 102.0, 51.0, 52.0],
            "high": [100.0, 102.0, 51.0, 52.0],
            "low": [100.0, 102.0, 51.0, 52.0],
            "close": [100.0, 102.0, 51.0, 52.0],
            "volume": [1000.0, 1000.0, 4000.0, 4000.0],
        }
    )


def test_split_adjusted_series_removes_discontinuity_and_ledger_updates() -> None:
    bars = _bars()
    ca = [
        {
            "symbol": "AAA",
            "action_type": "SPLIT",
            "ex_date": dt.date(2025, 1, 3),
            "params_json": {"split_factor": 2.0},
        }
    ]
    out = adjust_prices(
        "AAA", bars, dt.date(2025, 1, 1), dt.date(2025, 1, 4), method="ca", corporate_actions=ca
    )
    # day2 adjusted close = 102 / 2 = 51, continuous with day3 close=51
    assert out.loc[out["date"] == dt.date(2025, 1, 2), "close"].iloc[0] == pytest.approx(51.0)
    assert out.loc[out["date"] == dt.date(2025, 1, 2), "volume"].iloc[0] == pytest.approx(2000.0)

    ledger = apply_ca_to_position(
        qty_before=100,
        avg_cost_before=40.0,
        ex_date_price=51.0,
        action=CorporateActionEvent(
            symbol="AAA", action_type="SPLIT", ex_date=dt.date(2025, 1, 3), params={"split_factor": 2.0}
        ),
    )
    assert ledger["qty_after"] == pytest.approx(200)
    assert ledger["avg_cost_after"] == pytest.approx(20.0)


def test_cash_dividend_cash_credit_on_pay_date_with_withholding() -> None:
    action = CorporateActionEvent(
        symbol="AAA",
        action_type="CASH_DIVIDEND",
        ex_date=dt.date(2025, 2, 1),
        pay_date=dt.date(2025, 2, 20),
        params={"cash_per_share": 1000.0},
    )
    ledger = apply_ca_to_position(
        qty_before=10,
        avg_cost_before=20.0,
        ex_date_price=19.0,
        action=action,
    )
    assert ledger["cash_posting_date"] == dt.date(2025, 2, 20)
    assert ledger["tax"] == pytest.approx(10 * 1000 * WITHHOLDING_TAX_RATE)
    assert ledger["cash_delta"] == pytest.approx(10 * 1000 * (1 - WITHHOLDING_TAX_RATE))


def test_rights_issue_terp_and_fee_tax_applied() -> None:
    action = CorporateActionEvent(
        symbol="AAA",
        action_type="RIGHTS_ISSUE",
        ex_date=dt.date(2025, 3, 1),
        params={"ratio": 0.2, "subscription_price": 8000.0},
    )
    ledger = apply_ca_to_position(
        qty_before=100,
        avg_cost_before=10_000.0,
        ex_date_price=12_000.0,
        action=action,
        fee_rate=0.001,
        sell_tax_rate=0.001,
    )
    assert ledger["notes_json"]["terp"] == pytest.approx((12_000 + 0.2 * 8000) / 1.2)
    assert ledger["notes_json"]["right_value"] == pytest.approx(12_000 - (12_000 + 0.2 * 8000) / 1.2)
    assert ledger["fee"] > 0
    assert ledger["tax"] > 0


def test_total_return_series_matches_manual_example() -> None:
    bars = pd.DataFrame(
        {
            "date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3)],
            "open": [100.0, 102.0, 101.0],
            "high": [100.0, 102.0, 101.0],
            "low": [100.0, 102.0, 101.0],
            "close": [100.0, 102.0, 101.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )
    ca = [
        {
            "symbol": "AAA",
            "action_type": "CASH_DIVIDEND",
            "ex_date": dt.date(2025, 1, 2),
            "pay_date": dt.date(2025, 1, 3),
            "params_json": {"cash_per_share": 2.0},
        }
    ]
    out = adjust_prices(
        "AAA",
        bars,
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 3),
        method="ca",
        corporate_actions=ca,
        total_return=True,
    )
    # day1=1.0; day2=102/100=1.02; day3=(101+2)/102 * 1.02 = 1.03
    assert out["tr_index"].tolist() == pytest.approx([1.0, 1.02, 1.03])


def test_pit_ignore_ca_until_public_date() -> None:
    bars = _bars()
    ca = [
        {
            "symbol": "AAA",
            "action_type": "SPLIT",
            "ex_date": dt.date(2025, 1, 3),
            "public_date": dt.date(2025, 1, 5),
            "params_json": {"split_factor": 2.0},
        }
    ]
    out_before = adjust_prices(
        "AAA",
        bars,
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 4),
        method="ca",
        corporate_actions=ca,
        as_of_date=dt.date(2025, 1, 4),
    )
    out_after = adjust_prices(
        "AAA",
        bars,
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 4),
        method="ca",
        corporate_actions=ca,
        as_of_date=dt.date(2025, 1, 5),
    )
    assert out_before.loc[out_before["date"] == dt.date(2025, 1, 2), "close"].iloc[0] == pytest.approx(102.0)
    assert out_after.loc[out_after["date"] == dt.date(2025, 1, 2), "close"].iloc[0] == pytest.approx(51.0)
