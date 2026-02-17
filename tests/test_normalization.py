import pandas as pd
from core.normalization import (
    neutralize_by_group,
    neutralize_by_size,
    winsorize_series,
    zscore_cross_section,
)


def test_normalization_functions_basic() -> None:
    s = pd.Series([1, 2, 3, 100])
    w = winsorize_series(s)
    assert w.max() <= 100

    df = pd.DataFrame({"date": [1, 1, 2, 2], "value": [1.0, 2.0, 1.0, 3.0]})
    z = zscore_cross_section(df)
    assert len(z) == 4

    g = neutralize_by_group(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series(["A", "A", "B", "B"]))
    assert round(float(g.iloc[0] + g.iloc[1]), 8) == 0.0

    ns = neutralize_by_size(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([10, 20, 30, 40]))
    assert len(ns) == 4
