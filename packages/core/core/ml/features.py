from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from core.calendar_vn import get_trading_calendar_vn
from core.market_rules import load_market_rules

FEATURE_VERSION = "v1"
HORIZON = 21


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_ml_features(prices: pd.DataFrame, factors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build fixed feature set per spec with offline-safe defaults."""
    req = {"symbol", "timestamp", "close", "high", "low", "volume", "value_vnd"}
    missing = req - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = prices.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    for n in [1, 5, 21, 63, 126, 252]:
        df[f"ret_{n}d"] = g["close"].pct_change(n)
    df["rev_5d"] = -df["ret_5d"]

    rets = g["close"].pct_change()
    for n in [20, 60, 120]:
        df[f"vol_{n}d"] = rets.groupby(df["symbol"]).rolling(n).std().reset_index(level=0, drop=True)

    df["ema20"] = g["close"].transform(lambda s: _ema(s, 20))
    df["ema50"] = g["close"].transform(lambda s: _ema(s, 50))
    df["ema200"] = g["close"].transform(lambda s: _ema(s, 200))
    atr14 = g[["high", "low", "close"]].apply(_atr).reset_index(level=0, drop=True)
    df["atr14_pct"] = atr14 / df["close"].replace(0, np.nan)

    df["adv20_value"] = g["value_vnd"].rolling(20).mean().reset_index(level=0, drop=True)
    df["adv20_vol"] = g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    rules = load_market_rules("configs/market_rules_vn.yaml")
    df["spread_proxy"] = df["close"].map(lambda x: rules.get_tick_size(float(x), instrument="stock") if pd.notna(x) else np.nan) / df["close"].replace(0, np.nan)

    df["ema20_gt_ema50"] = (df["ema20"] > df["ema50"]).astype(float)
    df["close_gt_ema50"] = (df["close"] > df["ema50"]).astype(float)
    ema12 = g["close"].transform(lambda s: _ema(s, 12))
    ema26 = g["close"].transform(lambda s: _ema(s, 26))
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.groupby(df["symbol"]).transform(lambda s: _ema(s, 9))
    df["rsi14"] = g["close"].transform(lambda s: _rsi(s, 14))

    for name in ["value_score_z", "quality_score_z", "momentum_score_z", "lowvol_score_z", "dividend_score_z"]:
        df[name] = 0.0
    if factors is not None and not factors.empty:
        f = factors.copy()
        f["as_of_date"] = pd.to_datetime(f["as_of_date"]).dt.date
        pivot = f.pivot_table(index=["symbol", "as_of_date"], columns="factor", values="score", aggfunc="last").reset_index()
        df["as_of_date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df.merge(pivot, how="left", on=["symbol", "as_of_date"])
        for c, k in [
            ("value", "value_score_z"),
            ("quality", "quality_score_z"),
            ("momentum", "momentum_score_z"),
            ("low_vol", "lowvol_score_z"),
            ("dividend", "dividend_score_z"),
        ]:
            if c in df:
                df[k] = df[c].fillna(0.0)

    if "sector" not in df:
        df["sector"] = "OTHER"
    if "exchange" not in df:
        df["exchange"] = "HOSE"
    top15 = df["sector"].value_counts().head(15).index
    df["sector_norm"] = np.where(df["sector"].isin(top15), df["sector"], "OTHER")
    df = pd.get_dummies(df, columns=["sector_norm", "exchange"], prefix=["sector", "exchange"]) 

    cal = get_trading_calendar_vn()
    df["as_of_date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["realized_date"] = df["as_of_date"].map(lambda d: cal.shift_trading_days(d, HORIZON))

    future_close = df[["symbol", "as_of_date", "close"]].rename(
        columns={"as_of_date": "realized_date", "close": "close_t_plus_h"}
    )
    df = df.merge(future_close, on=["symbol", "realized_date"], how="left")
    df["y"] = df["close_t_plus_h"] / df["close"] - 1.0

    vn = df.groupby("as_of_date", as_index=False)["close"].mean().sort_values("as_of_date")
    vn["realized_date"] = vn["as_of_date"].map(lambda d: cal.shift_trading_days(d, HORIZON))
    vn_future = vn[["as_of_date", "close"]].rename(
        columns={"as_of_date": "realized_date", "close": "close_t_plus_h"}
    )
    vn = vn.merge(vn_future, on="realized_date", how="left")
    vn["vn_ret_h"] = vn["close_t_plus_h"] / vn["close"] - 1.0
    df = df.merge(vn[["as_of_date", "vn_ret_h"]], on="as_of_date", how="left")
    df["y_excess"] = df["y"] - df["vn_ret_h"]

    # Point-in-time cross-sectional median impute by date
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        med = df.groupby("as_of_date")[c].transform("median")
        df[c] = df[c].fillna(med)

    df["feature_version"] = FEATURE_VERSION
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "symbol", "timestamp", "as_of_date", "realized_date", "close_t_plus_h", "y", "y_excess", "feature_version", "vn_ret_h"
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def rebalance_dates(dates: pd.Series, freq: int = 21) -> list[dt.date]:
    ds = sorted(pd.to_datetime(dates).dt.date.unique())
    return [ds[i] for i in range(0, len(ds), freq)]
