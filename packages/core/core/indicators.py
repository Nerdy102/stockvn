from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def ema_incremental(price: float, prev_ema: float | None, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    if prev_ema is None:
        return float(price)
    return alpha * float(price) + (1.0 - alpha) * float(prev_ema)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)


@dataclass
class RSIState:
    avg_gain: float = 0.0
    avg_loss: float = 0.0
    prev_close: float | None = None
    warmup_count: int = 0


def rsi_incremental(price: float, state: RSIState, window: int = 14) -> tuple[float, RSIState]:
    if state.prev_close is None:
        state.prev_close = float(price)
        return 0.0, state

    delta = float(price) - float(state.prev_close)
    gain = max(delta, 0.0)
    loss = max(-delta, 0.0)

    if state.warmup_count < window:
        state.avg_gain = (state.avg_gain * state.warmup_count + gain) / (state.warmup_count + 1)
        state.avg_loss = (state.avg_loss * state.warmup_count + loss) / (state.warmup_count + 1)
        state.warmup_count += 1
    else:
        state.avg_gain = (state.avg_gain * (window - 1) + gain) / window
        state.avg_loss = (state.avg_loss * (window - 1) + loss) / window

    state.prev_close = float(price)
    if state.avg_loss == 0:
        return 100.0 if state.avg_gain > 0 else 0.0, state
    rs = state.avg_gain / state.avg_loss
    return 100.0 - (100.0 / (1.0 + rs)), state


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    pv = typical * volume
    cum_pv = pv.cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).ffill()


@dataclass
class RollingMeanState:
    window: int
    buf: Deque[float]
    total: float

    @classmethod
    def new(cls, window: int) -> "RollingMeanState":
        return cls(window=window, buf=deque(), total=0.0)


def rolling_mean_incremental(x: float, state: RollingMeanState) -> tuple[float | None, RollingMeanState]:
    state.buf.append(float(x))
    state.total += float(x)
    if len(state.buf) > state.window:
        state.total -= state.buf.popleft()
    if len(state.buf) < state.window:
        return None, state
    return state.total / state.window, state


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA20"] = sma(out["close"], 20)
    out["EMA20"] = ema(out["close"], 20)
    out["EMA50"] = ema(out["close"], 50)
    out["RSI14"] = rsi(out["close"], 14)
    line, sig, hist = macd(out["close"])
    out["MACD"] = line
    out["MACD_SIGNAL"] = sig
    out["MACD_HIST"] = hist
    out["ATR14"] = atr(out["high"], out["low"], out["close"], 14)
    out["VWAP"] = vwap(out["high"], out["low"], out["close"], out["volume"])
    return out
