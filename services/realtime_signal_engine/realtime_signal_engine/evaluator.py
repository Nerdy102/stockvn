from __future__ import annotations

from typing import Any

import pandas as pd

from core.signals.dsl import evaluate


def evaluate_setups(history: pd.DataFrame, indicators: dict[str, float]) -> dict[str, bool]:
    if history.empty:
        return {"trend": False, "breakout": False, "volume_spike": False}
    close = float(history.iloc[-1]["close"])
    vol = float(history.iloc[-1]["volume"])

    ema20 = float(indicators.get("EMA20", 0.0))
    ema50 = float(indicators.get("EMA50", 0.0))
    trend = bool(ema20 > ema50 and close > ema50)

    look = history.tail(21)
    prev20 = look.iloc[:-1] if len(look) > 1 else look
    max_high20 = float(prev20["high"].max()) if not prev20.empty else float(history["high"].max())
    avg_vol20 = (
        float(prev20["volume"].mean()) if not prev20.empty else float(history["volume"].mean())
    )
    breakout = bool(close > max_high20 and vol > 1.5 * avg_vol20)
    volume_spike = bool(vol > 2.0 * avg_vol20)

    return {"trend": trend, "breakout": breakout, "volume_spike": volume_spike}


def evaluate_alert_dsl_on_bar_close(history: pd.DataFrame, expression: str) -> bool:
    out = evaluate(expression, history)
    return bool(out.iloc[-1]) if len(out) > 0 else False


def governance_paused_flag(config: dict[str, Any]) -> bool:
    return bool(config.get("governance_paused", False))
