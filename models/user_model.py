from __future__ import annotations

import pandas as pd


class UserModel:
    """Baseline model stub for users to replace with their own quant model."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Baseline does not train statefully; custom models can store fitted params.
        _ = (X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Simple baseline: prefer stronger 20-day momentum.
        score = X.get("mom20", pd.Series(index=X.index, dtype=float)).fillna(0.0)
        return score.astype(float)
