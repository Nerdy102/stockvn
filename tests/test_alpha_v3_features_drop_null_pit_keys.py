from __future__ import annotations

import pandas as pd

from core.alpha_v3.features import _prepare_fundamentals_pti


def test_prepare_fundamentals_pti_drops_null_symbol_or_public_date() -> None:
    fundamentals = pd.DataFrame(
        [
            {"symbol": "AAA", "as_of_date": "2025-01-01", "period_end": "2024-12-31", "public_date": "2025-02-15"},
            {"symbol": None, "as_of_date": "2025-01-01", "period_end": "2024-12-31", "public_date": "2025-02-15"},
            {"symbol": "BBB", "as_of_date": "2025-01-01", "period_end": None, "public_date": None},
        ]
    )

    out = _prepare_fundamentals_pti(fundamentals)
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "AAA"
