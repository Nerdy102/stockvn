from __future__ import annotations

import numpy as np
import pandas as pd


def detect_regime(close: float, ema20: float, ema50: float) -> str:
    if close > ema50 and ema20 > ema50:
        return "risk_on"
    return "risk_off"


def compute_confidence_v2(
    *,
    has_min_rows: bool,
    close: float,
    volume: float,
    dollar_vol_lookback: pd.Series,
    atr_pct: float,
    model_id: str,
    regime: str,
) -> tuple[int, str, list[str], dict[str, float | str]]:
    tags: list[str] = []

    data_score = 30 if has_min_rows else 0

    if len(dollar_vol_lookback.dropna()) < 60:
        liquidity_score = 10
    else:
        p20 = float(np.percentile(dollar_vol_lookback.dropna().tail(60), 20))
        p80 = float(np.percentile(dollar_vol_lookback.dropna().tail(60), 80))
        dollar_vol = close * volume
        if volume <= 0 or dollar_vol < p20:
            liquidity_score = 0
        elif dollar_vol >= p80:
            liquidity_score = 20
        else:
            liquidity_score = 10
    if liquidity_score == 0:
        tags.append("Thanh khoản thấp")

    if atr_pct < 0.003:
        volatility_score = 0
    elif atr_pct <= 0.06:
        volatility_score = 10
    else:
        volatility_score = 0
        tags.append("Biến động cao")

    if model_id == "model_1":
        regime_score = 30 if regime == "risk_on" else 0
    elif model_id == "model_2":
        regime_score = 15 if regime == "risk_off" else 5
    else:
        regime_score = 30 if regime == "risk_on" else 0

    total = data_score + liquidity_score + volatility_score + regime_score
    bucket = "Cao" if total >= 70 else "Vừa" if total >= 45 else "Thấp"
    debug = {
        "data_score": data_score,
        "liquidity_score": liquidity_score,
        "volatility_score": volatility_score,
        "regime_score": regime_score,
        "total_score": total,
    }
    return total, bucket, tags, debug
