from __future__ import annotations

import pandas as pd

from core.signals.dsl import evaluate


def evaluate_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    out = evaluate(expr, df)
    return out.fillna(False).astype(bool)
