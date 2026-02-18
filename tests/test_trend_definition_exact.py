from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "realtime_signal_engine"))

from realtime_signal_engine.evaluator import evaluate_setups


def test_trend_definition_exact() -> None:
    history = pd.DataFrame(
        [
            {"close": 10.0, "high": 10.0, "low": 10.0, "volume": 100},
            {"close": 11.0, "high": 11.0, "low": 11.0, "volume": 120},
        ]
    )
    setups = evaluate_setups(history, {"EMA20": 10.5, "EMA50": 10.0})
    assert setups["trend"] is True

    setups2 = evaluate_setups(history, {"EMA20": 9.9, "EMA50": 10.0})
    assert setups2["trend"] is False
