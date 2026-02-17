from __future__ import annotations

import pandas as pd
from core.signals.dsl import evaluate, parse


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10, 10, 11, 12, 13, 14],
            "high": [11, 11, 12, 13, 14, 15],
            "low": [9, 9, 10, 11, 12, 13],
            "close": [10, 10, 11, 12, 13, 14],
            "volume": [100, 120, 130, 200, 300, 500],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )


def test_parse_smoke() -> None:
    node = parse("CROSSOVER(close, SMA(2)) AND volume > 1.5*AVG(volume, 2)")
    assert node is not None


def test_evaluate_bool() -> None:
    df = _df()
    out = evaluate("SMA(2) > 0", df)
    assert out.dtype == bool


def test_crossover_runs() -> None:
    df = _df()
    out = evaluate("CROSSOVER(close, SMA(3))", df)
    assert len(out) == len(df)
