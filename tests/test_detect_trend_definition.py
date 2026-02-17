import pandas as pd

from core.technical import detect_trend


def test_detect_trend_definition() -> None:
    idx = pd.date_range('2025-01-01', periods=80, freq='D')
    close = pd.Series([100 + i * 0.4 for i in range(80)], index=idx)
    df = pd.DataFrame({'open': close * 0.99, 'high': close * 1.01, 'low': close * 0.98, 'close': close, 'volume': 1_000_000}, index=idx)
    assert detect_trend(df) is True
