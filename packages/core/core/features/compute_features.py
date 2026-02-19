from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame, as_of_ts: object | None = None) -> pd.DataFrame:
    w = df.copy()
    ts_col = "timestamp" if "timestamp" in w.columns else "date"
    if as_of_ts is not None:
        w = w[pd.to_datetime(w[ts_col], errors="coerce") <= pd.to_datetime(as_of_ts)]
    w = w.reset_index(drop=True)
    w["ema20"] = _ema(w["close"], 20)
    w["ema50"] = _ema(w["close"], 50)
    w["rsi14"] = _rsi(w["close"], 14)
    w["atr14"] = _atr(w, 14)
    w["vol_avg20"] = w["volume"].rolling(20).mean()
    w["high20_prev"] = w["high"].rolling(20).max().shift(1)
    w["low20_prev"] = w["low"].rolling(20).min().shift(1)
    w["breakout_up"] = w["close"] > w["high20_prev"]
    w["breakout_dn"] = w["close"] < w["low20_prev"]
    return w
