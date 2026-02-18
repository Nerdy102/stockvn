import pandas as pd
from core.monitoring.data_quality import compute_data_quality_metrics
from core.monitoring.drift import compute_weekly_drift_metrics


def test_data_quality_metrics_basic() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "timeframe": ["1D", "1D", "1D"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 200, 300],
            "value_vnd": [1000, 2000, 3000],
        }
    )
    out = compute_data_quality_metrics(df, "csv", "1D")
    assert any(m["metric_name"] == "ohlc_invariants_violation_rate" for m in out)


def test_drift_metrics_shape() -> None:
    r = pd.Series([0.001] * 20 + [0.01] * 20)
    v = pd.Series([100] * 20 + [500] * 20)
    s = pd.Series([0.001] * 40)
    f = pd.Series([1.0] * 40)
    out = compute_weekly_drift_metrics(r, v, s, f)
    assert len(out) == 4
