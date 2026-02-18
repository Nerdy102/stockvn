from __future__ import annotations

import pandas as pd

from core.ml.diagnostics import ic_decay


def test_ic_decay_handles_existing_y_excess_without_duplicate_column_ambiguity() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "as_of_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
            "score_final": [0.1, 0.2, 0.3, 0.4],
            "y_excess": [0.01, 0.02, -0.01, 0.03],
        }
    )
    out = ic_decay(df, horizons=[1])
    assert "ic_decay_1" in out
