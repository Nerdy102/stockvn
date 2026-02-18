from __future__ import annotations

import pandas as pd

from core.alpha_v3.features import build_ml_features_v3


def test_point_in_time_fundamentals_public_date() -> None:
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-02-01", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100, "value_vnd": 1000},
            {"symbol": "AAA", "timestamp": "2024-02-20", "open": 10, "high": 10, "low": 10, "close": 11, "volume": 100, "value_vnd": 1100},
            {"symbol": "VNINDEX", "timestamp": "2024-02-01", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000, "value_vnd": 100000},
            {"symbol": "VNINDEX", "timestamp": "2024-02-20", "open": 100, "high": 100, "low": 100, "close": 101, "volume": 1000, "value_vnd": 101000},
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "as_of_date": "2023-12-31",
                "period_end": "2023-12-31",
                "public_date": "2024-02-15",
                "public_date_is_assumed": False,
                "revenue_ttm_vnd": 123,
            }
        ]
    )
    out = build_ml_features_v3(prices=prices, fundamentals=fundamentals)
    out = out[out["symbol"] == "AAA"].sort_values("date")
    assert str(out.iloc[0]["fundamental_effective_public_date"]) in {"NaT", "None"}
    assert str(out.iloc[1]["fundamental_effective_public_date"]) == "2024-02-15"


def test_point_in_time_fundamentals_assumed_public_date_by_statement_type() -> None:
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-04-15", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100, "value_vnd": 1000},
            {"symbol": "AAA", "timestamp": "2024-05-10", "open": 10, "high": 10, "low": 10, "close": 11, "volume": 100, "value_vnd": 1100},
            {"symbol": "AAA", "timestamp": "2024-06-15", "open": 10, "high": 10, "low": 10, "close": 12, "volume": 100, "value_vnd": 1200},
            {"symbol": "VNINDEX", "timestamp": "2024-04-15", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000, "value_vnd": 100000},
            {"symbol": "VNINDEX", "timestamp": "2024-05-10", "open": 100, "high": 100, "low": 100, "close": 101, "volume": 1000, "value_vnd": 101000},
            {"symbol": "VNINDEX", "timestamp": "2024-06-15", "open": 100, "high": 100, "low": 100, "close": 102, "volume": 1000, "value_vnd": 102000},
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "as_of_date": "2024-03-31",
                "period_end": "2024-03-31",
                "public_date": None,
                "statement_type": "quarterly",
            },
            {
                "symbol": "AAA",
                "as_of_date": "2023-12-31",
                "period_end": "2023-12-31",
                "public_date": None,
                "statement_type": "annual",
            },
        ]
    )
    out = build_ml_features_v3(prices=prices, fundamentals=fundamentals)
    aaa = out[out["symbol"] == "AAA"].sort_values("date").reset_index(drop=True)
    assert str(aaa.loc[0, "fundamental_effective_public_date"]) in {"NaT", "None"}
    assert str(aaa.loc[1, "fundamental_effective_public_date"]) == "2024-04-29"
    assert float(aaa.loc[1, "fundamental_public_date_limitation_flag"]) == 1.0
