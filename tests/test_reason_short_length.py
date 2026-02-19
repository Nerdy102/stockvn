from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def test_reason_short_length() -> None:
    rows = []
    for i in range(220):
        rows.append({
            "date": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)),
            "open": 100 + i,
            "high": 101 + i,
            "low": 99 + i,
            "close": 100 + i,
            "volume": 1_000_000,
        })
    sig = run_signal("model_1", "FPT", "1D", pd.DataFrame(rows))
    assert len(sig.reason_short) <= 140
