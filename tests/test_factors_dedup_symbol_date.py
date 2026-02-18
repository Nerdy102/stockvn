from __future__ import annotations

import pandas as pd

from core.factors import compute_factors


def test_compute_factors_handles_duplicate_symbol_date_rows() -> None:
    tickers = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "Tech",
                "is_bank": 0,
                "is_broker": 0,
                "shares_outstanding": 1_000_000,
            }
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "Tech",
                "net_income_ttm_vnd": 10_000_000,
                "equity_vnd": 100_000_000,
                "total_assets_vnd": 200_000_000,
                "net_debt_vnd": 0.0,
                "ebitda_ttm_vnd": 1_000_000.0,
                "cfo_ttm_vnd": 1_000_000.0,
                "dividends_ttm_vnd": 100_000.0,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2025-01-02", "symbol": "AAA", "close": 10.0, "volume": 1000, "value_vnd": 10_000},
            {"date": "2025-01-02", "symbol": "AAA", "close": 10.5, "volume": 1500, "value_vnd": 15_000},
            {"date": "2025-01-03", "symbol": "AAA", "close": 11.0, "volume": 900, "value_vnd": 9_900},
        ]
    )

    out = compute_factors(tickers=tickers, fundamentals=fundamentals, prices=prices)
    assert not out.scores.empty
    assert "AAA" in out.raw_metrics.index
    assert out.raw_metrics.loc["AAA", "market_cap"] == 11_000_000.0
