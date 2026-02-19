from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def _sample_df() -> pd.DataFrame:
    rows = []
    base = 100.0
    for i in range(80):
        close = base + i * 0.4
        rows.append(
            {
                "date": f"2025-01-{(i%28)+1:02d}",
                "open": close - 0.3,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1_000_000 + i * 1000,
            }
        )
    return pd.DataFrame(rows)


def test_all_models_have_reason_short() -> None:
    df = _sample_df()
    for model_id in ["model_1", "model_2", "model_3"]:
        sig = run_signal(model_id, "FPT", "1D", df)
        assert sig.reason_short.strip()
        assert len(sig.reason_short) <= 140
        assert sig.confidence_bucket in {"Thấp", "Vừa", "Cao"}
        assert len(sig.risk_tags) <= 2
