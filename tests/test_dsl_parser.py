import pandas as pd
from core.signals.dsl import evaluate


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1, 1, 1, 1],
            "high": [1, 2, 3, 4],
            "low": [1, 1, 1, 1],
            "close": [1, 1, 2, 3],
            "volume": [1, 1, 1, 1],
        }
    )


def test_precedence_not_and_or() -> None:
    df = _df()
    a = evaluate("NOT (close > 2) OR close > 2 AND close > 10", df)
    b = evaluate("(NOT (close > 2)) OR ((close > 2) AND (close > 10))", df)
    assert (a == b).all()


def test_crossover_semantics() -> None:
    df = _df()
    out = evaluate("CROSSOVER(close, SMA(2))", df)
    assert bool(out.iloc[2]) is True
