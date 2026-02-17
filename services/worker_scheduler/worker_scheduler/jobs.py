from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from sqlmodel import Session, select

from core.db.models import (
    AlertEvent,
    AlertRule,
    FactorScore,
    Fundamental,
    IndicatorValue,
    PriceOHLCV,
    Signal,
    Ticker,
)
from core.factors import compute_factors
from core.indicators import add_indicators
from core.signals.dsl import evaluate
from core.settings import Settings
from core.technical import detect_breakout, detect_pullback, detect_trend, detect_volume_spike
from data.etl.ingest import ingest_fundamentals, ingest_prices, ingest_tickers
from data.providers.base import BaseMarketDataProvider

log = logging.getLogger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert to float; turn NaN/Inf/None into default."""
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _json_sanitize(obj: Any) -> Any:
    """
    Ensure JSON-serializable dict has no NaN/Inf (Postgres JSON rejects token NaN).
    Convert numpy scalars -> python scalars; NaN/Inf -> None.
    """
    # Convert numpy scalar types -> python scalar
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            obj = obj.item()
    except Exception:
        pass

    if obj is None:
        return None

    # floats: remove NaN/Inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # pandas NA / NaT or other scalar missing values
    try:
        miss = pd.isna(obj)
        if isinstance(miss, bool) and miss:
            return None
    except Exception:
        pass

    # datetime/date -> iso string (safe for JSON)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    return obj


def _provider_last_date(provider: BaseMarketDataProvider, symbols: List[str]) -> Optional[dt.date]:
    """Find last available 1D date from demo/provider (prefer VNINDEX)."""
    prefer = ["VNINDEX"] + [s for s in symbols if s != "VNINDEX"]
    for sym in prefer:
        try:
            df = provider.get_ohlcv(sym, "1D")
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df["date"].max()
        except Exception:
            continue
    return None


def ensure_seeded(session: Session, provider: BaseMarketDataProvider, settings: Settings) -> None:
    """Seed DB with demo data + compute jobs (idempotent)."""
    log.info("Seeding database (idempotent)...")

    ingest_tickers(session, provider.get_tickers())
    ingest_fundamentals(session, provider.get_fundamentals())

    tickers = provider.get_tickers()
    symbols = tickers["symbol"].tolist()

    # Ingest prices 1D
    for sym in symbols:
        df = provider.get_ohlcv(sym, "1D")
        if not df.empty:
            ingest_prices(session, sym, "1D", df)

    # FIXED: intraday seed range should use last_date from demo data (not dt.date.today()).
    last_date = _provider_last_date(provider, symbols) or dt.date.today()
    start = last_date - dt.timedelta(days=20)

    # Ingest intraday bars (MVP) to support 60m/15m charting
    for sym in symbols:
        if sym == "VNINDEX":
            continue
        for tf in ["60m", "15m"]:
            df_i = provider.get_intraday(sym, tf, start=start, end=last_date)
            if df_i is not None and not df_i.empty:
                ingest_prices(session, sym, tf, df_i)

    update_market_caps(session)
    compute_indicators(session)
    compute_factor_scores(session)
    compute_technical_setups(session)
    ensure_demo_alert_rules(session)
    generate_alerts(session)

    log.info("Seed done.")


def ingest_prices_job(session: Session, provider: BaseMarketDataProvider) -> None:
    """Job: ingest_prices (idempotent). For CSV it's stable."""
    tickers = provider.get_tickers()
    for sym in tickers["symbol"].tolist():
        df = provider.get_ohlcv(sym, "1D")
        if not df.empty:
            ingest_prices(session, sym, "1D", df)
    update_market_caps(session)


def update_market_caps(session: Session) -> None:
    """Update tickers.market_cap based on latest 1D close (idempotent)."""
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return
    pdf = pd.DataFrame([p.model_dump() for p in prices])
    pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date
    last_close = pdf.sort_values(["symbol", "date"]).groupby("symbol")["close"].last()

    tickers = session.exec(select(Ticker)).all()
    for t in tickers:
        if t.symbol in last_close.index and t.shares_outstanding > 0:
            t.market_cap = float(last_close.loc[t.symbol]) * float(t.shares_outstanding)
            session.merge(t)
    session.commit()


def compute_indicators(session: Session) -> None:
    """Job: compute_indicators -> store IndicatorValue (idempotent)."""
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"])

    indicators = ["SMA20", "EMA20", "EMA50", "RSI14", "MACD", "ATR14", "VWAP"]

    for sym, g in df.groupby("symbol"):
        if sym == "VNINDEX":
            continue
        g = g.set_index("timestamp")[["open", "high", "low", "close", "volume"]].tail(260)
        if g.empty:
            continue
        ind = add_indicators(g)
        ind = ind.dropna().tail(200)
        for ts, row in ind.iterrows():
            for name in indicators:
                if name in row and pd.notna(row[name]):
                    session.merge(
                        IndicatorValue(
                            symbol=sym,
                            timeframe="1D",
                            timestamp=ts.to_pydatetime(),
                            name=name,
                            value=float(row[name]),
                        )
                    )
    session.commit()
    log.info("Indicators computed (MVP).")


def compute_factor_scores(session: Session) -> None:
    """Job: compute_factor_scores -> FactorScore (idempotent)."""
    tickers = session.exec(select(Ticker)).all()
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    fundamentals = session.exec(select(Fundamental)).all()

    if not tickers or not prices or not fundamentals:
        log.warning("Not enough data to compute factors.")
        return

    tdf = pd.DataFrame([t.model_dump() for t in tickers if t.symbol != "VNINDEX"])
    fdf = pd.DataFrame([f.model_dump() for f in fundamentals])
    pdf = pd.DataFrame([p.model_dump() for p in prices])
    pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date

    out = compute_factors(
        tickers=tdf[["symbol", "sector", "is_bank", "is_broker", "shares_outstanding"]].copy(),
        fundamentals=fdf.copy(),
        prices=pdf[["date", "symbol", "close", "volume", "value_vnd"]].copy(),
        benchmark_symbol="VNINDEX",
    )

    as_of = pdf["date"].max()
    for sym, row in out.scores.iterrows():
        raw: Dict[str, Any] = out.raw_metrics.loc[sym].to_dict() if sym in out.raw_metrics.index else {}
        raw = _json_sanitize(raw) or {}

        for factor in ["value", "quality", "momentum", "low_vol", "dividend"]:
            session.merge(
                FactorScore(
                    symbol=sym,
                    as_of_date=as_of,
                    factor=factor,
                    score=_safe_float(row.get(factor), 0.0),
                    raw=raw,
                )
            )

        total = (
            _safe_float(row.get("value"), 0.0)
            + _safe_float(row.get("quality"), 0.0)
            + _safe_float(row.get("momentum"), 0.0)
        )
        session.merge(
            FactorScore(
                symbol=sym,
                as_of_date=as_of,
                factor="total",
                score=total,
                raw=raw,
            )
        )

    session.commit()
    log.info("Factor scores computed (MVP).")


def compute_technical_setups(session: Session) -> None:
    """Job: compute_technical_setups -> Signal (idempotent)."""
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"])

    for sym, g in df.groupby("symbol"):
        if sym == "VNINDEX":
            continue
        g = g.set_index("timestamp")[["open", "high", "low", "close", "volume"]].tail(260)
        if g.empty:
            continue
        ts = g.index[-1].to_pydatetime()
        if detect_breakout(g):
            session.merge(Signal(symbol=sym, timeframe="1D", timestamp=ts, signal_type="breakout", strength=1.0, meta={}))
        if detect_trend(g):
            session.merge(Signal(symbol=sym, timeframe="1D", timestamp=ts, signal_type="trend", strength=1.0, meta={}))
        if detect_pullback(g):
            session.merge(Signal(symbol=sym, timeframe="1D", timestamp=ts, signal_type="pullback", strength=1.0, meta={}))
        if detect_volume_spike(g):
            session.merge(Signal(symbol=sym, timeframe="1D", timestamp=ts, signal_type="volume_spike", strength=1.0, meta={}))
    session.commit()
    log.info("Technical setups computed (MVP).")


def ensure_demo_alert_rules(session: Session) -> None:
    if session.exec(select(AlertRule)).first() is not None:
        return
    demo_path = Path("configs/alerts/demo_rules.yaml")
    if not demo_path.exists():
        return
    data = yaml.safe_load(demo_path.read_text(encoding="utf-8")) or {}
    for r in data.get("rules", []) or []:
        symbols_csv = ",".join(r.get("symbols", []) or [])
        session.add(
            AlertRule(
                name=str(r["name"]),
                timeframe=str(r.get("timeframe", "1D")),
                expression=str(r["expression"]),
                symbols_csv=symbols_csv,
                is_active=True,
            )
        )
    session.commit()


def generate_alerts(session: Session) -> None:
    """Job: generate_alerts by evaluating DSL rules on latest 1D bars (idempotent)."""
    rules = session.exec(select(AlertRule).where(AlertRule.is_active)).all()
    if not rules:
        return
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return

    df = pd.DataFrame([p.model_dump() for p in prices])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"])

    for rule in rules:
        symbols: List[str]
        if rule.symbols_csv.strip():
            symbols = [s.strip() for s in rule.symbols_csv.split(",") if s.strip()]
        else:
            symbols = sorted(df["symbol"].unique().tolist())

        for sym in symbols:
            if sym == "VNINDEX":
                continue
            g = df[df["symbol"] == sym].set_index("timestamp")[["open", "high", "low", "close", "volume"]].tail(300)
            if g.empty:
                continue
            try:
                out = evaluate(rule.expression, g)
                trig = bool(out.iloc[-1])
            except Exception as e:
                log.warning("Rule eval failed: %s (%s)", rule.name, e)
                continue

            if trig:
                ts = g.index[-1].to_pydatetime()
                msg = f"Rule '{rule.name}' triggered for {sym} at {ts.isoformat()}."
                session.merge(
                    AlertEvent(
                        rule_id=int(rule.id or 0),
                        symbol=sym,
                        triggered_at=ts,
                        message=msg,
                        meta={"expression": rule.expression},
                    )
                )
    session.commit()
    log.info("Alerts generated (MVP).")
