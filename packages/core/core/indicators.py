from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP cumulative."""
    typical = (high + low + close) / 3.0
    pv = typical * volume
    cum_pv = pv.cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).ffill()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append common indicators to OHLCV DataFrame."""
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
