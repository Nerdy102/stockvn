from __future__ import annotations

import pandas as pd
from core.execution_model import ExecutionAssumptions, slippage_bps
from core.factors import compute_factors
from core.regime import REGIME_RISK_OFF, REGIME_TREND_UP, classify_market_regime


def test_regime_classification_basic() -> None:
    idx = pd.date_range("2025-01-01", periods=80, freq="D")
    up = pd.Series(range(1000, 1080), index=idx)
    regime_up = classify_market_regime(up)
    assert regime_up.iloc[-1] == REGIME_TREND_UP

    down = pd.Series(range(1080, 1000, -1), index=idx)
    regime_down = classify_market_regime(down)
    assert regime_down.iloc[-1] == REGIME_RISK_OFF


def test_slippage_bounds_monotonic() -> None:
    a = ExecutionAssumptions(base_slippage_bps=10, k1_participation=50, k2_volatility=40)
    low = slippage_bps(order_notional=1e8, adtv=1e11, atr_pct=0.01, assumptions=a)
    high = slippage_bps(order_notional=5e9, adtv=1e10, atr_pct=0.05, assumptions=a)
    assert low >= 0
    assert high > low


def test_factor_negative_denominator_handling() -> None:
    tickers = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "Tech",
                "is_bank": 0,
                "is_broker": 0,
                "shares_outstanding": 1_000_000,
            },
            {
                "symbol": "BBB",
                "sector": "Tech",
                "is_bank": 0,
                "is_broker": 0,
                "shares_outstanding": 1_000_000,
            },
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "Tech",
                "net_income_ttm_vnd": -10,
                "equity_vnd": 100,
                "total_assets_vnd": 200,
                "cfo_ttm_vnd": 10,
                "net_debt_vnd": 10,
                "ebitda_ttm_vnd": 10,
                "dividends_ttm_vnd": 1,
            },
            {
                "symbol": "BBB",
                "sector": "Tech",
                "net_income_ttm_vnd": 10,
                "equity_vnd": -100,
                "total_assets_vnd": 200,
                "cfo_ttm_vnd": 10,
                "net_debt_vnd": 10,
                "ebitda_ttm_vnd": 10,
                "dividends_ttm_vnd": 1,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2025-01-01", "symbol": "AAA", "close": 10, "volume": 100, "value_vnd": 1000},
            {"date": "2025-01-01", "symbol": "BBB", "close": 10, "volume": 100, "value_vnd": 1000},
        ]
    )
    out = compute_factors(tickers, fundamentals, prices)
    assert pd.isna(out.raw_metrics.loc["AAA", "PE"])  # NI <= 0 invalid for PE
    assert pd.isna(out.raw_metrics.loc["BBB", "PB"])  # Equity <= 0 invalid for PB
