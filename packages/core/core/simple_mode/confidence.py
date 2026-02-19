from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def detect_regime(close: float, ema20: float, ema50: float) -> str:
    if any(np.isnan([close, ema20, ema50])):
        return "unknown"
    if close > ema50 and ema20 > ema50:
        return "risk_on"
    if close < ema50 or ema20 < ema50:
        return "risk_off"
    return "unknown"


def liquidity_bucket(close: float, volume: float, dv_lookback: pd.Series) -> str:
    if volume <= 0 or dv_lookback.empty or len(dv_lookback.dropna()) < 60:
        return "unknown"
    dv = close * volume
    p20 = float(dv_lookback.quantile(0.2))
    p80 = float(dv_lookback.quantile(0.8))
    if dv < p20:
        return "thấp"
    if dv >= p80:
        return "cao"
    return "vừa"


def confidence_bucket(
    *,
    has_min_rows: bool,
    liquidity: str,
    regime: str,
    regime_expected: str,
    atr_pct: float,
) -> tuple[str, int]:
    score = 0
    if has_min_rows:
        score += 1
    if liquidity != "thấp":
        score += 1
    if regime == regime_expected:
        score += 1
    if 0.008 <= atr_pct <= 0.06:
        score += 1
    if score >= 3:
        return "Cao", score
    if score == 2:
        return "Vừa", score
    return "Thấp", score


def build_risk_tags(
    *, liquidity: str, regime: str, atr_pct: float, has_min_rows: bool
) -> list[str]:
    tags: list[str] = []
    if liquidity == "thấp":
        tags.append("Thanh khoản thấp")
    if regime == "risk_off":
        tags.append("Chế độ rủi ro-xấu (risk-off)")
    if atr_pct > 0.06:
        tags.append("Biến động cao")
    if not has_min_rows:
        tags.append("Thiếu dữ liệu")
    return tags[:2]


def compact_debug_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = None
            else:
                out[k] = round(v, 6)
        else:
            out[k] = v
    return out
