from __future__ import annotations

import pandas as pd

from core.indicators import ema

REGIME_TREND_UP = "trend_up"
REGIME_SIDEWAY = "sideway"
REGIME_RISK_OFF = "risk_off"


def classify_market_regime(vnindex_close: pd.Series) -> pd.Series:
    c = pd.to_numeric(vnindex_close, errors="coerce").dropna()
    if c.empty:
        return pd.Series(dtype=object)
    e20 = ema(c, 20)
    e50 = ema(c, 50)
    spread = (e20 - e50) / e50.replace(0, pd.NA)

    regime = pd.Series(REGIME_SIDEWAY, index=c.index, dtype=object)
    regime[(e20 > e50) & (c > e50) & (spread > 0.002)] = REGIME_TREND_UP
    regime[(c < e50) & (spread < -0.002)] = REGIME_RISK_OFF
    return regime


def regime_exposure_multiplier(regime: str) -> float:
    r = str(regime)
    if r == REGIME_TREND_UP:
        return 1.0
    if r == REGIME_SIDEWAY:
        return 0.75
    if r == REGIME_RISK_OFF:
        return 0.4
    return 0.7
