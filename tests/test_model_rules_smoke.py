from __future__ import annotations

import pandas as pd

from core.simple_mode.models import run_signal


def _df() -> pd.DataFrame:
    rows = []
    for i in range(240):
        c = 100 + i * 0.2
        rows.append(
            {
                "date": str(pd.Timestamp("2025-02-01") + pd.Timedelta(days=i))[:10],
                "open": c - 0.5,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 1_000_000 + i * 500,
            }
        )
    return pd.DataFrame(rows)


def test_models_have_required_fields() -> None:
    df = _df()
    for m in ["model_1", "model_2", "model_3"]:
        sig = run_signal(m, "FPT", "1D", df)
        assert sig.reason_short
        assert sig.confidence_bucket in {"Thấp", "Vừa", "Cao"}
        assert len(sig.risk_tags) <= 2
        for k in [
            "ema20",
            "ema50",
            "rsi14",
            "atr14",
            "atr_pct",
            "vol",
            "vol_avg20",
            "high20_prev",
            "close",
        ]:
            assert k in sig.debug_fields
