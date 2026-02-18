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
