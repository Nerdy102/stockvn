from __future__ import annotations

import pandas as pd

from research.strategies import user_strategy_adapter


def generate_weights(frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    return user_strategy_adapter.generate_weights(frame, universe)
