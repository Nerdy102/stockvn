import pandas as pd

from scripts.run_eval_lab import _simulate


def test_no_lookahead_execution_tplus1() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1, 1, 1],
        }
    )
    w = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    eq, _, _ = _simulate(
        df, w, commission_bps=0.0, sell_tax_bps=0.0, slippage_bps=0.0, turnover_cap=1.0
    )
    # first executable return is from day1 signal to day2 execution->day2 close (0 here), then day2->day3
    assert len(eq) == 2
