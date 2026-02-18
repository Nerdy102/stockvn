from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from core.market_rules import load_market_rules

FEATURE_VERSION = "v3"
HORIZON = 21

BASE_FEATURES_V3 = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "rev_5d",
    "vol_20d", "vol_60d", "vol_120d",
    "atr14_pct",
    "adv20_value", "adv20_vol",
    "spread_proxy",
    "limit_hit_20d",
    "rsi14", "macd_hist",
    "ema20_gt_ema50", "close_gt_ema50", "ema50_slope",
    "value_score_z", "quality_score_z", "momentum_score_z", "lowvol_score_z", "dividend_score_z",
    "regime_trend_up", "regime_sideways", "regime_risk_off",
    "net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util",
    "imb_1_day", "imb_3_day", "spread_day",
    "rv_day", "vol_first_hour_ratio",
    "fundamental_public_date_is_assumed",
]


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


def _prepare_factor_pivot(factors: pd.DataFrame | None) -> pd.DataFrame:
    if factors is None or factors.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    f = factors.copy()
    f["date"] = pd.to_datetime(f["as_of_date"]).dt.date
    pivot = f.pivot_table(index=["symbol", "date"], columns="factor", values="score", aggfunc="last").reset_index()
    out = pd.DataFrame({"symbol": pivot["symbol"], "date": pivot["date"]})
    out["value_score_z"] = pivot.get("value", 0.0)
    out["quality_score_z"] = pivot.get("quality", 0.0)
    out["momentum_score_z"] = pivot.get("momentum", 0.0)
    out["lowvol_score_z"] = pivot.get("low_vol", 0.0)
    out["dividend_score_z"] = pivot.get("dividend", 0.0)
    return out


def _prepare_fundamentals_pti(fundamentals: pd.DataFrame | None) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=["symbol", "effective_public_date"])
    f = fundamentals.copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).dt.date
    f["period_end"] = pd.to_datetime(f.get("period_end"), errors="coerce").dt.date
    f["public_date"] = pd.to_datetime(f.get("public_date"), errors="coerce").dt.date
    assumed = f.get("public_date_is_assumed", False)
    if isinstance(assumed, bool):
        assumed = pd.Series([assumed] * len(f), index=f.index)
    f["public_date_is_assumed"] = assumed.fillna(False).astype(bool)

    missing_public = f["public_date"].isna() & f["period_end"].notna()
    f.loc[missing_public, "public_date"] = f.loc[missing_public, "period_end"] + pd.to_timedelta(45, unit="D")
    f.loc[missing_public, "public_date_is_assumed"] = True
    f["effective_public_date"] = f["public_date"]

    cols = [
        "symbol",
        "effective_public_date",
        "public_date_is_assumed",
        "revenue_ttm_vnd",
        "net_income_ttm_vnd",
        "gross_profit_ttm_vnd",
        "ebitda_ttm_vnd",
        "cfo_ttm_vnd",
        "total_assets_vnd",
        "total_liabilities_vnd",
        "equity_vnd",
        "net_debt_vnd",
    ]
    for c in cols:
        if c not in f.columns:
            f[c] = np.nan
    return f[cols].sort_values(["symbol", "effective_public_date"])


def _asof_merge_pti(base: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    if pit.empty:
        out = base.copy()
        out["fundamental_public_date_is_assumed"] = False
        out["fundamental_effective_public_date"] = pd.NaT
        return out
    merged = pd.merge_asof(
        base.assign(_date_ts=pd.to_datetime(base["date"])).sort_values("_date_ts"),
        pit.assign(_pub_ts=pd.to_datetime(pit["effective_public_date"])).sort_values("_pub_ts"),
        by="symbol",
        left_on="_date_ts",
        right_on="_pub_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["fundamental_public_date_is_assumed"] = merged["public_date_is_assumed"].astype("boolean").fillna(False).astype(float)
    merged["fundamental_effective_public_date"] = pd.to_datetime(merged["effective_public_date"], errors="coerce").dt.date
    return merged.drop(columns=["_date_ts", "_pub_ts"], errors="ignore")


def add_regime_flags(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    vn = out[out["symbol"] == "VNINDEX"].copy()
    if vn.empty:
        out["regime_trend_up"] = 0.0
        out["regime_sideways"] = 1.0
        out["regime_risk_off"] = 0.0
        return out

    vn = vn[["date", "ema20", "ema50", "close"]].drop_duplicates("date").sort_values("date")
    cond_trend = (vn["ema20"] > vn["ema50"]) & (vn["close"] > vn["ema50"]) & ((vn["ema50"] - vn["ema50"].shift(10)) > 0)
    cond_off = (vn["close"] < vn["ema50"]) & (vn["ema20"] < vn["ema50"])
    vn["regime_trend_up"] = cond_trend.astype(float)
    vn["regime_risk_off"] = cond_off.astype(float)
    vn["regime_sideways"] = ((vn["regime_trend_up"] == 0) & (vn["regime_risk_off"] == 0)).astype(float)
    out = out.merge(vn[["date", "regime_trend_up", "regime_sideways", "regime_risk_off"]], on="date", how="left")
    return out


def assert_no_leakage(features_date: dt.date, label_date: dt.date, horizon: int = HORIZON) -> None:
    min_label_date = features_date + dt.timedelta(days=horizon)
    if label_date < min_label_date:
        raise RuntimeError(
            "Leakage guard triggered: "
            f"label_date={label_date} is earlier than min allowed {min_label_date} "
            f"for features_date={features_date}, horizon={horizon}."
        )


def enforce_no_leakage_guard(
    df: pd.DataFrame,
    feature_date_col: str = "date",
    label_date_col: str = "label_date",
    horizon: int = HORIZON,
) -> None:
    for _, row in df.iterrows():
        assert_no_leakage(
            pd.to_datetime(row[feature_date_col]).date(),
            pd.to_datetime(row[label_date_col]).date(),
            horizon=horizon,
        )


def assert_feature_timestamps_not_future(df: pd.DataFrame, date_col: str = "date") -> None:
    if "timestamp" not in df.columns:
        return
    ts_date = pd.to_datetime(df["timestamp"]).dt.date
    feat_date = pd.to_datetime(df[date_col]).dt.date
    bad = df[ts_date > feat_date]
    if not bad.empty:
        sample = bad[["symbol", date_col, "timestamp"]].head(3).to_dict("records")
        raise RuntimeError(f"Feature timestamp leakage detected: {sample}")


def build_ml_features_v3(
    prices: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    flow_features: pd.DataFrame | None = None,
    orderbook_features: pd.DataFrame | None = None,
    intraday_features: pd.DataFrame | None = None,
    tickers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    req = {"symbol", "timestamp", "close", "high", "low", "volume", "value_vnd"}
    missing = req - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = prices.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
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
    df["spread_proxy"] = df["close"].map(
        lambda x: rules.get_tick_size(float(x), instrument="stock") if pd.notna(x) else np.nan
    ) / df["close"].replace(0, np.nan)

    if {"ceiling_price", "floor_price"}.issubset(df.columns):
        limit_hit = ((df["close"] >= df["ceiling_price"]) | (df["close"] <= df["floor_price"])).astype(float)
    else:
        limit_hit = pd.Series(0.0, index=df.index)
    df["limit_hit_20d"] = limit_hit.groupby(df["symbol"]).rolling(20).sum().reset_index(level=0, drop=True)

    ema12 = g["close"].transform(lambda s: _ema(s, 12))
    ema26 = g["close"].transform(lambda s: _ema(s, 26))
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.groupby(df["symbol"]).transform(lambda s: _ema(s, 9))
    df["rsi14"] = g["close"].transform(lambda s: _rsi(s, 14))

    df["ema20_gt_ema50"] = (df["ema20"] > df["ema50"]).astype(float)
    df["close_gt_ema50"] = (df["close"] > df["ema50"]).astype(float)
    df["ema50_slope"] = (df["ema50"] - g["ema50"].shift(10)) / 10.0

    factor_pivot = _prepare_factor_pivot(factors)
    df = df.merge(factor_pivot, on=["symbol", "date"], how="left")
    for c in ["value_score_z", "quality_score_z", "momentum_score_z", "lowvol_score_z", "dividend_score_z"]:
        if c not in df.columns:
            df[c] = np.nan

    pit_f = _prepare_fundamentals_pti(fundamentals)
    df = _asof_merge_pti(df, pit_f)

    df = add_regime_flags(df)

    def _merge_daily(src: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
        if src is None or src.empty:
            return pd.DataFrame(columns=["symbol", "date", *cols])
        out = src.copy()
        out["date"] = pd.to_datetime(out.get("date", out.get("as_of_date"))).dt.date
        keep = ["symbol", "date", *[c for c in cols if c in out.columns]]
        return out[keep]

    flow = _merge_daily(flow_features, ["net_foreign_val_5d", "net_foreign_val_20d", "foreign_flow_intensity", "foreign_room_util"])
    orderbook = _merge_daily(orderbook_features, ["imb_1_day", "imb_3_day", "spread_day"])
    intra = _merge_daily(intraday_features, ["rv_day", "vol_first_hour_ratio"])

    df = df.merge(flow, on=["symbol", "date"], how="left")
    df = df.merge(orderbook, on=["symbol", "date"], how="left")
    df = df.merge(intra, on=["symbol", "date"], how="left")

    if tickers is not None and not tickers.empty:
        meta = tickers[["symbol", "sector", "exchange"]].drop_duplicates("symbol")
        df = df.merge(meta, on="symbol", how="left")
    if "sector" not in df.columns:
        df["sector"] = "OTHER"
    df["sector"] = df["sector"].fillna("OTHER")
    if "exchange" not in df.columns:
        df["exchange"] = "HOSE"
    df["exchange"] = df["exchange"].fillna("HOSE")
    top15 = df["sector"].value_counts().head(15).index
    df["sector_norm"] = np.where(df["sector"].isin(top15), df["sector"], "OTHER")
    df = pd.get_dummies(df, columns=["sector_norm", "exchange"], prefix=["sector", "exchange"])

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        med = df.groupby("date")[c].transform("median")
        df[c] = df[c].fillna(med)
        df[c] = df[c].fillna(0.0)

    assert_feature_timestamps_not_future(df, date_col="date")

    for c in BASE_FEATURES_V3:
        if c not in df.columns:
            df[c] = 0.0

    selected = [
        "symbol",
        "timestamp",
        "date",
        "fundamental_effective_public_date",
        *BASE_FEATURES_V3,
        *[c for c in df.columns if c.startswith("sector_") or c.startswith("exchange_")],
    ]
    df = df[selected].copy()
    df["feature_version"] = FEATURE_VERSION
    return df


def feature_columns_v3(df: pd.DataFrame) -> list[str]:
    prefixes = ("sector_", "exchange_")
    cols = [c for c in BASE_FEATURES_V3 if c in df.columns]
    cols.extend([c for c in df.columns if c.startswith(prefixes)])
    return cols
