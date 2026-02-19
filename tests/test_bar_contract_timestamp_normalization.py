from __future__ import annotations

import datetime as dt

from data.contracts.bars import BarRecord, display_timestamp_for_market, normalize_bar_timestamp


def test_bar_contract_timestamp_normalization_utc() -> None:
    ts = normalize_bar_timestamp("2025-01-01 09:15:00+07:00")
    assert ts.tzinfo is not None
    assert ts.utcoffset() == dt.timedelta(0)

    bar = BarRecord(
        symbol="FPT",
        market="vn",
        timeframe="1D",
        ts=dt.datetime(2025, 1, 1, 2, 15, 0),
        open=1,
        high=1,
        low=1,
        close=1,
        volume=1,
    )
    assert bar.ts.tzinfo is not None
    assert bar.ts.utcoffset() == dt.timedelta(0)


def test_display_timestamp_for_market_vn_vs_crypto() -> None:
    ts_utc = dt.datetime(2025, 1, 1, 2, 0, 0, tzinfo=dt.timezone.utc)
    ts_vn = display_timestamp_for_market(ts_utc, "vn")
    ts_crypto = display_timestamp_for_market(ts_utc, "crypto")
    assert ts_vn.utcoffset() == dt.timedelta(hours=7)
    assert ts_crypto.utcoffset() == dt.timedelta(0)
