from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.indicators import add_indicators, atr


@dataclass(frozen=True)
class Zone:
    kind: str  # supply | demand
    start: pd.Timestamp
    end: pd.Timestamp
    low: float
    high: float


def detect_breakout(df: pd.DataFrame, lookback: int = 20, volume_multiple: float = 1.5) -> bool:
    """Breakout MVP: close > prior N-day high and volume spike."""
    d = df.sort_index()
    if d.shape[0] < lookback + 2:
        return False
    prior_high = d["high"].shift(1).rolling(lookback).max().iloc[-1]
    avg_vol = d["volume"].shift(1).rolling(lookback).mean().iloc[-1]
    last = d.iloc[-1]
    return bool(last["close"] > prior_high and last["volume"] > volume_multiple * avg_vol)


def detect_trend(df: pd.DataFrame) -> bool:
    """Trend MVP: EMA20 > EMA50 and close > EMA50."""
    d = add_indicators(df.sort_index())
    if d.shape[0] < 60:
        return False
    last = d.iloc[-1]
    return bool(last["EMA20"] > last["EMA50"] and last["close"] > last["EMA50"])


def detect_pullback(df: pd.DataFrame) -> bool:
    """Pullback MVP: uptrend and close near EMA20."""
    d = add_indicators(df.sort_index())
    if d.shape[0] < 60:
        return False
    last = d.iloc[-1]
    in_trend = bool(last["EMA20"] > last["EMA50"] and last["close"] > last["EMA50"])
    near = abs(last["close"] - last["EMA20"]) / max(1.0, last["close"]) < 0.02
    return bool(in_trend and near)


def detect_volume_spike(df: pd.DataFrame, lookback: int = 20, k: float = 2.0) -> bool:
    d = df.sort_index()
    if d.shape[0] < lookback + 1:
        return False
    avg_vol = d["volume"].shift(1).rolling(lookback).mean().iloc[-1]
    return bool(d["volume"].iloc[-1] > k * avg_vol)


def detect_supply_demand_zones(
    df: pd.DataFrame, pivot_window: int = 5, zone_atr_mult: float = 0.6
) -> list[Zone]:
    """Heuristic supply/demand zones using pivot highs/lows."""
    d = df.sort_index().copy()
    if d.shape[0] < pivot_window * 2 + 20:
        return []
    d["ATR14"] = atr(d["high"], d["low"], d["close"], 14)

    zones: list[Zone] = []
    highs = d["high"].values
    lows = d["low"].values
    idx = d.index

    for i in range(pivot_window, len(d) - pivot_window):
        win_high = highs[i - pivot_window : i + pivot_window + 1]
        win_low = lows[i - pivot_window : i + pivot_window + 1]
        atr_i = float(d["ATR14"].iloc[i]) if not np.isnan(d["ATR14"].iloc[i]) else 0.0
        height = atr_i * zone_atr_mult

        if highs[i] == win_high.max():
            zones.append(
                Zone(
                    kind="supply",
                    start=idx[i],
                    end=idx[-1],
                    low=float(highs[i] - height),
                    high=float(highs[i] + height),
                )
            )
        if lows[i] == win_low.min():
            zones.append(
                Zone(
                    kind="demand",
                    start=idx[i],
                    end=idx[-1],
                    low=float(lows[i] - height),
                    high=float(lows[i] + height),
                )
            )

    zones = sorted(zones, key=lambda z: z.start, reverse=True)[:6]
    return zones


def auto_trendline(df: pd.DataFrame, kind: str = "support", lookback: int = 60) -> dict[str, float]:
    """Auto trendline MVP via linear regression on lows/highs."""
    d = df.sort_index().tail(lookback)
    if d.shape[0] < 10:
        return {"slope": 0.0, "intercept": float("nan")}
    y = d["low"].values if kind == "support" else d["high"].values
    x = np.arange(len(d))
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}
