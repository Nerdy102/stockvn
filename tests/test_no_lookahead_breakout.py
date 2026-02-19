from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def test_breakout_uses_high20_prev_shifted() -> None:
    rows = []
    for i in range(230):
        high = 100 + i
        close = 95 + i
        rows.append(
            {
                "date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i))[:10],
                "open": close - 1,
                "high": high,
                "low": close - 2,
                "close": close,
                "volume": 1_000_000,
            }
        )
    df = pd.DataFrame(rows)
    sig = run_signal("model_1", "FPT", "1D", df)
    assert "high20_prev" in sig.debug_fields
    assert float(sig.debug_fields["high20_prev"]) <= float(sig.debug_fields["close"]) + 100
