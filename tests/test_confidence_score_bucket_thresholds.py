from __future__ import annotations

import pandas as pd

from core.simple_mode.confidence_v2 import compute_confidence_v2


def test_confidence_score_bucket_thresholds() -> None:
    s = pd.Series([1_000_000.0] * 60)
    score, bucket, _tags, _ = compute_confidence_v2(
        has_min_rows=True,
        close=100,
        volume=20_000,
        dollar_vol_lookback=s,
        atr_pct=0.02,
        model_id="model_1",
        regime="risk_on",
    )
    assert score >= 70
    assert bucket == "Cao"
