from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sqlmodel import Session

from core.db.models import Fundamental, PriceOHLCV, Ticker


def ingest_tickers(session: Session, tickers_df: pd.DataFrame) -> int:
    n = 0
    for _, r in tickers_df.iterrows():
        tags_raw = str(r.get("tags", "") or "").strip()
        tags: Dict[str, Any] = {}
        if tags_raw:
            tags["catalysts"] = [t.strip() for t in tags_raw.split(";") if t.strip()]

        t = Ticker(
            symbol=str(r["symbol"]),
            name=str(r["name"]),
            exchange=str(r["exchange"]),
            sector=str(r["sector"]),
            industry=str(r["industry"]),
            shares_outstanding=int(r.get("shares_outstanding", 0) or 0),
            market_cap=float(r.get("market_cap", 0.0) or 0.0),
            is_bank=bool(r.get("is_bank", False)),
            is_broker=bool(r.get("is_broker", False)),
            tags=tags,
        )
        session.merge(t)
        n += 1
    session.commit()
    return n


def ingest_fundamentals(session: Session, fundamentals_df: pd.DataFrame) -> int:
    n = 0
    df = fundamentals_df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    for _, r in df.iterrows():
        f = Fundamental(
            symbol=str(r["symbol"]),
            as_of_date=r["as_of_date"],
            sector=str(r["sector"]),
            is_bank=bool(r.get("is_bank", False)),
            is_broker=bool(r.get("is_broker", False)),
            revenue_ttm_vnd=float(r.get("revenue_ttm_vnd", 0.0) or 0.0),
            net_income_ttm_vnd=float(r.get("net_income_ttm_vnd", 0.0) or 0.0),
            gross_profit_ttm_vnd=float(r.get("gross_profit_ttm_vnd", 0.0) or 0.0),
            ebitda_ttm_vnd=float(r.get("ebitda_ttm_vnd", 0.0) or 0.0),
            cfo_ttm_vnd=float(r.get("cfo_ttm_vnd", 0.0) or 0.0),
            dividends_ttm_vnd=float(r.get("dividends_ttm_vnd", 0.0) or 0.0),
            total_assets_vnd=float(r.get("total_assets_vnd", 0.0) or 0.0),
            total_liabilities_vnd=float(r.get("total_liabilities_vnd", 0.0) or 0.0),
            equity_vnd=float(r.get("equity_vnd", 0.0) or 0.0),
            net_debt_vnd=float(r.get("net_debt_vnd", 0.0) or 0.0),
            nim=_to_float_or_none(r.get("nim")),
            casa=_to_float_or_none(r.get("casa")),
            cir=_to_float_or_none(r.get("cir")),
            npl_ratio=_to_float_or_none(r.get("npl_ratio")),
            llr_coverage=_to_float_or_none(r.get("llr_coverage")),
            credit_growth=_to_float_or_none(r.get("credit_growth")),
            car=_to_float_or_none(r.get("car")),
            margin_lending_vnd=_to_float_or_none(r.get("margin_lending_vnd")),
            adtv_sensitivity=_to_float_or_none(r.get("adtv_sensitivity")),
            proprietary_gains_ratio=_to_float_or_none(r.get("proprietary_gains_ratio")),
        )
        session.merge(f)
        n += 1
    session.commit()
    return n


def ingest_prices(session: Session, symbol: str, timeframe: str, ohlcv_df: pd.DataFrame) -> int:
    n = 0
    df = ohlcv_df.copy()
    if timeframe == "1D":
        df["timestamp"] = pd.to_datetime(df["date"]).dt.to_pydatetime()
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.to_pydatetime()

    for _, r in df.iterrows():
        p = PriceOHLCV(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=r["timestamp"],
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r.get("volume", 0.0) or 0.0),
            value_vnd=float(r.get("value_vnd", 0.0) or 0.0),
        )
        session.merge(p)
        n += 1
    session.commit()
    return n


def _to_float_or_none(x):
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None
