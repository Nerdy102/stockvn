from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorOutput:
    """Factor scores + raw metrics."""

    scores: pd.DataFrame
    raw_metrics: pd.DataFrame


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd == 0.0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (x - mu) / sd


def compute_factors(
    tickers: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_symbol: str = "VNINDEX",
) -> FactorOutput:
    """Compute factor scores (Value/Quality/Momentum/LowVol/Dividend).

    Factor definitions (MVP):
    - Value: PE low, PB low, earnings yield high
    - Quality:
        * Non-fin: ROE/ROA high, CFO/NI high, NetDebt/EBITDA low
        * Banks: CASA high, CIR low, NPL low, LLR coverage high, CAR high
    - Momentum: 3M/6M/12M returns
    - LowVol: 60D/120D volatility (lower better)
    - Dividend: dividend yield (TTM dividends / market cap)

    Inputs:
    - tickers: columns [symbol, sector, is_bank, is_broker, shares_outstanding]
    - fundamentals: snapshot per symbol
    - prices: columns [date, symbol, close, volume, value_vnd] for timeframe 1D
    """
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.date
    px = px.sort_values(["symbol", "date"])

    panel = px.pivot(index="date", columns="symbol", values="close").sort_index()
    rets = panel.pct_change()

    last_close = panel.iloc[-1] if not panel.empty else pd.Series(dtype=float)

    uni = tickers.merge(
        fundamentals,
        on=["symbol", "sector"],
        how="left",
        suffixes=("", "_f"),
    ).set_index("symbol")

    uni["last_close"] = last_close.reindex(uni.index)
    uni["market_cap"] = uni["last_close"] * pd.to_numeric(uni["shares_outstanding"], errors="coerce").fillna(0.0)

    mc = uni["market_cap"].replace(0, np.nan)
    ni = pd.to_numeric(uni.get("net_income_ttm_vnd"), errors="coerce").replace(0, np.nan)
    eq = pd.to_numeric(uni.get("equity_vnd"), errors="coerce").replace(0, np.nan)
    assets = pd.to_numeric(uni.get("total_assets_vnd"), errors="coerce").replace(0, np.nan)

    # Value
    uni["PE"] = mc / ni
    uni["PB"] = mc / eq
    uni["EARNINGS_YIELD"] = ni / mc

    # Quality (common)
    uni["ROE"] = ni / eq
    uni["ROA"] = ni / assets
    uni["CFO_TO_NI"] = pd.to_numeric(uni.get("cfo_ttm_vnd"), errors="coerce") / ni

    # Non-fin leverage
    uni["NET_DEBT_TO_EBITDA"] = pd.to_numeric(uni.get("net_debt_vnd"), errors="coerce") / pd.to_numeric(
        uni.get("ebitda_ttm_vnd"), errors="coerce"
    ).replace(0, np.nan)

    # Bank metrics
    uni["BANK_NIM"] = pd.to_numeric(uni.get("nim"), errors="coerce")
    uni["BANK_CASA"] = pd.to_numeric(uni.get("casa"), errors="coerce")
    uni["BANK_CIR"] = pd.to_numeric(uni.get("cir"), errors="coerce")
    uni["BANK_NPL"] = pd.to_numeric(uni.get("npl_ratio"), errors="coerce")
    uni["BANK_LLR"] = pd.to_numeric(uni.get("llr_coverage"), errors="coerce")
    uni["BANK_CAR"] = pd.to_numeric(uni.get("car"), errors="coerce")

    # Dividend
    uni["DIV_YIELD"] = pd.to_numeric(uni.get("dividends_ttm_vnd"), errors="coerce") / mc

    # Momentum
    def _mom(sym: str, n: int) -> float:
        s = panel.get(sym)
        if s is None:
            return float("nan")
        s = s.dropna()
        if len(s) <= n:
            return float("nan")
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0)

    symbols = [c for c in panel.columns if c != benchmark_symbol]
    uni["MOM_3M"] = pd.Series({sym: _mom(sym, 63) for sym in symbols})
    uni["MOM_6M"] = pd.Series({sym: _mom(sym, 126) for sym in symbols})
    uni["MOM_12M"] = pd.Series({sym: _mom(sym, 252) for sym in symbols})

    # Low vol
    uni["VOL_60D"] = rets.rolling(60).std(ddof=0).iloc[-1].reindex(uni.index) if not rets.empty else np.nan
    uni["VOL_120D"] = rets.rolling(120).std(ddof=0).iloc[-1].reindex(uni.index) if not rets.empty else np.nan

    # Scores (higher better)
    value_score = _zscore(-uni["PE"]) + _zscore(-uni["PB"]) + _zscore(uni["EARNINGS_YIELD"])

    quality_non_fin = (
        _zscore(uni["ROE"])
        + _zscore(uni["ROA"])
        + _zscore(uni["CFO_TO_NI"])
        + _zscore(-uni["NET_DEBT_TO_EBITDA"])
    )

    quality_bank = (
        _zscore(uni["BANK_CASA"])
        + _zscore(-uni["BANK_CIR"])
        + _zscore(-uni["BANK_NPL"])
        + _zscore(uni["BANK_LLR"])
        + _zscore(uni["BANK_CAR"])
    )

    is_bank = pd.to_numeric(uni.get("is_bank"), errors="coerce").fillna(0).astype(int) == 1
    quality_score = quality_non_fin.where(~is_bank, other=quality_bank)

    momentum_score = _zscore(uni["MOM_3M"]) + _zscore(uni["MOM_6M"]) + _zscore(uni["MOM_12M"])
    lowvol_score = _zscore(-uni["VOL_60D"]) + _zscore(-uni["VOL_120D"])
    dividend_score = _zscore(uni["DIV_YIELD"])

    scores = pd.DataFrame(
        {
            "symbol": uni.index,
            "value": value_score,
            "quality": quality_score,
            "momentum": momentum_score,
            "low_vol": lowvol_score,
            "dividend": dividend_score,
        }
    ).set_index("symbol")

    raw_cols = [
        "PE",
        "PB",
        "EARNINGS_YIELD",
        "ROE",
        "ROA",
        "CFO_TO_NI",
        "NET_DEBT_TO_EBITDA",
        "BANK_NIM",
        "BANK_CASA",
        "BANK_CIR",
        "BANK_NPL",
        "BANK_LLR",
        "BANK_CAR",
        "DIV_YIELD",
        "MOM_3M",
        "MOM_6M",
        "MOM_12M",
        "VOL_60D",
        "VOL_120D",
        "market_cap",
    ]
    raw = uni[raw_cols].copy()

    return FactorOutput(scores=scores, raw_metrics=raw)
