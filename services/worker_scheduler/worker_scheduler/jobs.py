from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from core.db.models import (
    AlertEvent,
    AlertRule,
    BronzeFile,
    DataQualityMetric,
    DriftMetric,
    FactorScore,
    Fundamental,
    IndicatorState,
    IndicatorValue,
    JobRun,
    PriceOHLCV,
    Signal,
    Ticker,
)
from core.db.partitioning import ensure_partitions_monthly as ensure_partitions_monthly_impl
from core.factors import compute_factors
from core.indicators import RSIState, add_indicators, ema_incremental, rsi_incremental
from core.logging import get_logger
from core.monitoring.data_quality import compute_data_quality_metrics
from core.monitoring.drift import compute_weekly_drift_metrics
from core.settings import Settings
from core.signals.dsl import evaluate
from core.technical import detect_breakout, detect_pullback, detect_trend, detect_volume_spike
from data.bronze.writer import BronzeWriter
from data.etl.ingest import ingest_fundamentals, ingest_prices, ingest_tickers
from data.etl.pipeline import ingest_from_fixtures
from data.providers.base import BaseMarketDataProvider
from data.providers.ssi_fastconnect.mapper_stream import map_stream_payload
from data.repository.ssi_stream_ingest import SsiStreamIngestRepository
from sqlmodel import Session, select

log = get_logger(__name__)


def _create_redis_client(redis_url: str):
    from redis import Redis

    return Redis.from_url(redis_url, decode_responses=True)


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
        obj = obj

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
        miss = False

    # datetime/date -> iso string (safe for JSON)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    return obj


def _provider_last_date(provider: BaseMarketDataProvider, symbols: list[str]) -> dt.date | None:
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
        raw: dict[str, Any] = (
            out.raw_metrics.loc[sym].to_dict() if sym in out.raw_metrics.index else {}
        )
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
            session.merge(
                Signal(
                    symbol=sym,
                    timeframe="1D",
                    timestamp=ts,
                    signal_type="breakout",
                    strength=1.0,
                    meta={},
                )
            )
        if detect_trend(g):
            session.merge(
                Signal(
                    symbol=sym,
                    timeframe="1D",
                    timestamp=ts,
                    signal_type="trend",
                    strength=1.0,
                    meta={},
                )
            )
        if detect_pullback(g):
            session.merge(
                Signal(
                    symbol=sym,
                    timeframe="1D",
                    timestamp=ts,
                    signal_type="pullback",
                    strength=1.0,
                    meta={},
                )
            )
        if detect_volume_spike(g):
            session.merge(
                Signal(
                    symbol=sym,
                    timeframe="1D",
                    timestamp=ts,
                    signal_type="volume_spike",
                    strength=1.0,
                    meta={},
                )
            )
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
        symbols: list[str]
        if rule.symbols_csv.strip():
            symbols = [s.strip() for s in rule.symbols_csv.split(",") if s.strip()]
        else:
            symbols = sorted(df["symbol"].unique().tolist())

        for sym in symbols:
            if sym == "VNINDEX":
                continue
            g = (
                df[df["symbol"] == sym]
                .set_index("timestamp")[["open", "high", "low", "close", "volume"]]
                .tail(300)
            )
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


def ingest_ssi_fixtures_job(session: Session) -> dict[str, int]:
    """Offline ingest for SSI fixtures with bronze/silver/checkpoint."""
    return ingest_from_fixtures(session)


def bronze_retention_cleanup(session: Session) -> int:
    try:
        retention_days = int(os.getenv("BRONZE_RETENTION_DAYS", "30"))
    except ValueError:
        retention_days = 30
    retention_days = max(retention_days, 1)
    cutoff_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days)).date()
    rows = session.exec(select(BronzeFile).where(BronzeFile.date < cutoff_date)).all()
    deleted = 0
    for row in rows:
        try:
            Path(row.filepath).unlink(missing_ok=True)
        except Exception:
            log.warning("Failed deleting bronze file", extra={"filepath": row.filepath})
        session.delete(row)
        deleted += 1
    session.commit()
    return deleted


def cleanup_stream_dedup_job(session: Session) -> int:
    repo = SsiStreamIngestRepository(session)
    deleted = repo.cleanup_dedup_older_than_days(days=14)
    repo.commit()
    return deleted


def consume_ssi_stream_to_bronze_silver(session: Session) -> dict[str, int]:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    bronze_writers: dict[str, BronzeWriter] = {}
    consumer_name = os.getenv("SSI_STREAM_CONSUMER", "worker-1")
    stream_keys = [f"ssi:{name}" for name in ["X", "X-QUOTE", "X-TRADE", "R", "MI", "B", "F", "OL"]]
    redis_client = _create_redis_client(redis_url)

    for stream_key in stream_keys:
        try:
            redis_client.xgroup_create(stream_key, "silver_writer", id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    repo = SsiStreamIngestRepository(session)
    rows = redis_client.xreadgroup(
        groupname="silver_writer",
        consumername=consumer_name,
        streams={k: ">" for k in stream_keys},
        count=500,
        block=500,
    )

    processed = 0
    skipped = 0
    acked = 0
    seen_hashes: set[tuple[str, str]] = set()

    for stream_key, messages in rows:
        for msg_id, fields in messages:
            payload_raw = str(fields.get("payload") or "{}")
            rtype = str(fields.get("rtype") or stream_key.split(":", 1)[-1])
            channel = f"ssi_stream_{rtype}"
            payload_hash = repo.payload_hash(payload_raw)

            if (rtype, payload_hash) in seen_hashes or repo.is_duplicate(
                "ssi_stream", rtype, payload_hash
            ):
                redis_client.xack(stream_key, "silver_writer", msg_id)
                skipped += 1
                acked += 1
                continue

            try:
                payload_obj = json.loads(payload_raw)
                mapped_rtype, mapped_items = map_stream_payload(payload_obj)
                writer = bronze_writers.get(channel)
                if writer is None:
                    writer = BronzeWriter(
                        provider="ssi_fastconnect", channel=channel, session=session
                    )
                    bronze_writers[channel] = writer
                writer.write(
                    {
                        "rtype": mapped_rtype,
                        "payload": payload_obj,
                        "payload_hash": payload_hash,
                        "received_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
                repo.append_bronze(
                    channel=channel,
                    payload_hash=payload_hash,
                    payload=payload_raw,
                    rtype=mapped_rtype,
                )
                for kind, item in mapped_items:
                    if kind == "quote":
                        repo.upsert_quote(item)
                    elif kind == "trade":
                        repo.upsert_trade(item)
                    elif kind == "foreign_room":
                        repo.upsert_foreign_room(item)
                    elif kind == "index":
                        repo.upsert_index(item)
                    elif kind == "bar":
                        repo.upsert_bar(item)
                repo.mark_dedup("ssi_stream", rtype, payload_hash)
                repo.commit()
                seen_hashes.add((rtype, payload_hash))
                redis_client.xack(stream_key, "silver_writer", msg_id)
                acked += 1
                processed += 1
            except Exception:
                session.rollback()
                raise

    for writer in bronze_writers.values():
        writer.flush()

    return {"processed": processed, "skipped": skipped, "acked": acked}


def compute_indicators_incremental(session: Session, timeframe: str = "1D") -> int:
    """Incremental EMA20 + RSI14 updates; only processes rows newer than saved state."""
    prices = session.exec(
        select(PriceOHLCV)
        .where(PriceOHLCV.timeframe == timeframe)
        .order_by(PriceOHLCV.symbol, PriceOHLCV.timestamp)
    ).all()
    if not prices:
        return 0
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    updates = 0

    for sym, g in df.groupby("symbol"):
        ema_state_row = session.get(IndicatorState, (sym, timeframe, "EMA20"))
        rsi_state_row = session.get(IndicatorState, (sym, timeframe, "RSI14"))

        ema_prev = (
            float(ema_state_row.state_json.get("ema"))
            if ema_state_row and "ema" in ema_state_row.state_json
            else None
        )
        ema_last_ts = (
            pd.to_datetime(ema_state_row.state_json.get("last_ts"))
            if ema_state_row and ema_state_row.state_json.get("last_ts")
            else None
        )

        rsi_state = RSIState()
        rsi_last_ts = None
        if rsi_state_row:
            payload = rsi_state_row.state_json
            rsi_state = RSIState(
                avg_gain=float(payload.get("avg_gain", 0.0)),
                avg_loss=float(payload.get("avg_loss", 0.0)),
                prev_close=(
                    float(payload.get("prev_close"))
                    if payload.get("prev_close") is not None
                    else None
                ),
                warmup_count=int(payload.get("warmup_count", 0)),
            )
            rsi_last_ts = pd.to_datetime(payload.get("last_ts")) if payload.get("last_ts") else None

        for _, row in g.sort_values("timestamp").iterrows():
            ts = pd.to_datetime(row["timestamp"])
            if (
                ema_last_ts is not None
                and ts <= ema_last_ts
                and rsi_last_ts is not None
                and ts <= rsi_last_ts
            ):
                continue

            ema_prev = ema_incremental(float(row["close"]), ema_prev, span=20)
            rsi_val, rsi_state = rsi_incremental(float(row["close"]), rsi_state, window=14)

            session.merge(
                IndicatorValue(
                    symbol=sym,
                    timeframe=timeframe,
                    timestamp=ts.to_pydatetime(),
                    name="EMA20_INC",
                    value=float(ema_prev),
                )
            )
            session.merge(
                IndicatorValue(
                    symbol=sym,
                    timeframe=timeframe,
                    timestamp=ts.to_pydatetime(),
                    name="RSI14_INC",
                    value=float(rsi_val),
                )
            )
            updates += 1

            ema_last_ts = ts
            rsi_last_ts = ts

        session.merge(
            IndicatorState(
                symbol=sym,
                timeframe=timeframe,
                indicator_name="EMA20",
                state_json={
                    "ema": ema_prev,
                    "last_ts": ema_last_ts.isoformat() if ema_last_ts is not None else None,
                },
            )
        )
        session.merge(
            IndicatorState(
                symbol=sym,
                timeframe=timeframe,
                indicator_name="RSI14",
                state_json={
                    "avg_gain": rsi_state.avg_gain,
                    "avg_loss": rsi_state.avg_loss,
                    "prev_close": rsi_state.prev_close,
                    "warmup_count": rsi_state.warmup_count,
                    "last_ts": rsi_last_ts.isoformat() if rsi_last_ts is not None else None,
                },
            )
        )

    session.commit()
    return updates


def compute_data_quality_metrics_job(session: Session) -> list[dict[str, Any]]:
    run = JobRun(job_name="compute_data_quality_metrics")
    session.add(run)
    session.commit()

    prices = session.exec(select(PriceOHLCV)).all()
    if not prices:
        run.status = "completed"
        run.end_ts = dt.datetime.utcnow()
        session.add(run)
        session.commit()
        return []

    df = pd.DataFrame([p.model_dump() for p in prices])
    metrics = compute_data_quality_metrics(df, provider="db", timeframe="mixed")
    today = dt.date.today()
    for m in metrics:
        session.add(
            DataQualityMetric(
                metric_date=today,
                provider=str(m.get("provider", "db")),
                symbol=m.get("symbol"),
                timeframe=m.get("timeframe"),
                metric_name=str(m.get("metric_name")),
                metric_value=float(m.get("metric_value", 0.0)),
            )
        )
    run.status = "completed"
    run.end_ts = dt.datetime.utcnow()
    run.rows_in = len(df)
    run.rows_out = len(metrics)
    session.add(run)
    session.commit()
    return metrics


def compute_drift_metrics_job(session: Session) -> list[dict[str, Any]]:
    run = JobRun(job_name="compute_drift_metrics")
    session.add(run)
    session.commit()

    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        run.status = "completed"
        run.end_ts = dt.datetime.utcnow()
        session.add(run)
        session.commit()
        return []

    df = pd.DataFrame([p.model_dump() for p in prices]).sort_values(["symbol", "timestamp"])
    if df.empty:
        run.status = "completed"
        run.end_ts = dt.datetime.utcnow()
        session.add(run)
        session.commit()
        return []

    mkt = df[df["symbol"] == "VNINDEX"].copy()
    if mkt.empty:
        run.status = "completed"
        run.end_ts = dt.datetime.utcnow()
        session.add(run)
        session.commit()
        return []

    ret = mkt["close"].pct_change().dropna()
    vol = mkt["volume"].astype(float).iloc[1:]
    spread_proxy = (1.0 / mkt["close"].astype(float).replace(0, pd.NA)).fillna(0.0).iloc[1:]
    metrics = compute_weekly_drift_metrics(ret, vol, spread_proxy)

    today = dt.date.today()
    for m in metrics:
        session.add(
            DriftMetric(
                metric_date=today,
                metric_name=str(m.get("metric_name")),
                metric_value=float(m.get("metric_value", 0.0)),
                alert=bool(m.get("alert", False)),
            )
        )

    run.status = "completed"
    run.end_ts = dt.datetime.utcnow()
    run.rows_in = len(df)
    run.rows_out = len(metrics)
    session.add(run)
    session.commit()
    return metrics


def compute_daily_flow_features(session: Session) -> int:
    from core.db.models import ForeignRoom, MlFeature
    from core.ml.features_v2 import compute_foreign_flow_features

    rows = session.exec(select(ForeignRoom)).all()
    if not rows:
        return 0
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["as_of_date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = (
        df.sort_values(["symbol", "timestamp"])
        .groupby(["symbol", "as_of_date"], as_index=False)
        .last()
    )
    daily["net_foreign_val"] = daily["fbuy_val"].fillna(0.0) - daily["fsell_val"].fillna(0.0)
    adv = pd.DataFrame(
        [
            r.model_dump()
            for r in session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
        ]
    )
    if adv.empty:
        return 0
    adv["as_of_date"] = pd.to_datetime(adv["timestamp"]).dt.date
    adv = adv[["symbol", "as_of_date", "value_vnd"]].rename(columns={"value_vnd": "adv20_value"})
    ff = compute_foreign_flow_features(
        daily[["symbol", "as_of_date", "net_foreign_val", "current_room", "total_room"]],
        adv,
    )
    up = 0
    for _, r in ff.iterrows():
        obj = session.exec(
            select(MlFeature)
            .where(MlFeature.symbol == str(r["symbol"]))
            .where(MlFeature.as_of_date == r["as_of_date"])
            .where(MlFeature.feature_version == "v2")
        ).first()
        payload = {
            "net_foreign_val_5d": float(r.get("net_foreign_val_5d") or 0.0),
            "net_foreign_val_20d": float(r.get("net_foreign_val_20d") or 0.0),
            "foreign_flow_intensity": float(r.get("foreign_flow_intensity") or 0.0),
            "foreign_room_util": float(r.get("foreign_room_util") or 0.0),
        }
        if obj:
            obj.features_json.update(payload)
            session.add(obj)
        else:
            session.add(
                MlFeature(
                    symbol=str(r["symbol"]),
                    as_of_date=r["as_of_date"],
                    feature_version="v2",
                    features_json=payload,
                )
            )
        up += 1
    session.commit()
    return up


def compute_daily_orderbook_features(session: Session) -> int:
    from core.db.models import MlFeature, QuoteL2
    from core.ml.features_v2 import compute_orderbook_daily_features

    rows = session.exec(select(QuoteL2)).all()
    if not rows:
        return 0
    df = pd.DataFrame([r.model_dump() for r in rows])
    ob = compute_orderbook_daily_features(df)
    up = 0
    for _, r in ob.iterrows():
        obj = session.exec(
            select(MlFeature)
            .where(MlFeature.symbol == str(r["symbol"]))
            .where(MlFeature.as_of_date == r["as_of_date"])
            .where(MlFeature.feature_version == "v2")
        ).first()
        payload = {
            "imb_1_day": float(r.get("imb_1_day") or 0.0),
            "imb_3_day": float(r.get("imb_3_day") or 0.0),
            "spread_day": float(r.get("spread_day") or 0.0),
        }
        if obj:
            obj.features_json.update(payload)
            session.add(obj)
        else:
            session.add(
                MlFeature(
                    symbol=str(r["symbol"]),
                    as_of_date=r["as_of_date"],
                    feature_version="v2",
                    features_json=payload,
                )
            )
        up += 1
    session.commit()
    return up


def compute_intraday_daily_features(session: Session) -> int:
    from core.db.models import MlFeature
    from core.ml.features_v2 import compute_intraday_daily_features as _compute

    bars = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1m")).all()
    if not bars:
        return 0
    df = pd.DataFrame([r.model_dump() for r in bars])
    intr = _compute(df)
    up = 0
    for _, r in intr.iterrows():
        obj = session.exec(
            select(MlFeature)
            .where(MlFeature.symbol == str(r["symbol"]))
            .where(MlFeature.as_of_date == r["as_of_date"])
            .where(MlFeature.feature_version == "v2")
        ).first()
        payload = {
            "rv_day": float(r.get("rv_day") or 0.0),
            "vol_first_hour_ratio": float(r.get("vol_first_hour_ratio") or 0.0),
        }
        if obj:
            obj.features_json.update(payload)
            session.add(obj)
        else:
            session.add(
                MlFeature(
                    symbol=str(r["symbol"]),
                    as_of_date=r["as_of_date"],
                    feature_version="v2",
                    features_json=payload,
                )
            )
        up += 1
    session.commit()
    return up


def compute_ml_labels_rank_z(session: Session) -> int:
    from core.db.models import MlFeature
    from core.ml.features import build_ml_features
    from core.ml.targets import compute_rank_z_label

    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return 0
    df = build_ml_features(pd.DataFrame([p.model_dump() for p in prices]))
    df = compute_rank_z_label(df, col="y_excess")
    up = 0
    for _, r in df.dropna(subset=["y_rank_z"]).iterrows():
        obj = session.exec(
            select(MlFeature)
            .where(MlFeature.symbol == str(r["symbol"]))
            .where(MlFeature.as_of_date == r["as_of_date"])
            .where(MlFeature.feature_version == "v2")
        ).first()
        payload = {"y_rank_z": float(r["y_rank_z"]), "y_excess": float(r.get("y_excess", 0.0))}
        if obj:
            obj.features_json.update(payload)
            session.add(obj)
        else:
            session.add(
                MlFeature(
                    symbol=str(r["symbol"]),
                    as_of_date=r["as_of_date"],
                    feature_version="v2",
                    features_json=payload,
                )
            )
        up += 1
    session.commit()
    return up


def train_models_v2(session: Session) -> int:
    from core.db.models import ForeignRoom, MlPrediction, QuoteL2
    from core.ml.features import build_ml_features, feature_columns
    from core.ml.features_v2 import build_features_v2
    from core.ml.models_v2 import MlModelV2Bundle
    from core.ml.targets import compute_rank_z_label

    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return 0
    feat = build_ml_features(pd.DataFrame([p.model_dump() for p in prices]))
    feat = compute_rank_z_label(feat, col="y_excess")
    fr = pd.DataFrame([r.model_dump() for r in session.exec(select(ForeignRoom)).all()])
    if not fr.empty:
        fr["as_of_date"] = pd.to_datetime(fr["timestamp"]).dt.date
        fr = fr.groupby(["symbol", "as_of_date"], as_index=False).last()
        fr["net_foreign_val"] = fr["fbuy_val"].fillna(0.0) - fr["fsell_val"].fillna(0.0)
    quotes = pd.DataFrame([r.model_dump() for r in session.exec(select(QuoteL2)).all()])
    intraday = pd.DataFrame(
        [
            r.model_dump()
            for r in session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1m")).all()
        ]
    )
    feat = build_features_v2(
        feat,
        None,
        fr if not fr.empty else None,
        quotes if not quotes.empty else None,
        intraday if not intraday.empty else None,
    )
    feat = feat.dropna(subset=["y_rank_z"])
    cols = feature_columns(feat)
    model = MlModelV2Bundle().fit(feat[cols], feat["y_rank_z"])
    comp = model.predict_components(feat[cols])
    up = 0
    for i, (_, r) in enumerate(feat.iterrows()):
        old = session.exec(
            select(MlPrediction)
            .where(MlPrediction.model_id == "ensemble_v2")
            .where(MlPrediction.symbol == str(r["symbol"]))
            .where(MlPrediction.as_of_date == r["as_of_date"])
        ).first()
        per_model = {
            "ridge_rank_v2": float(comp["ridge_rank_v2"][i]),
            "hgbr_rank_v2": float(comp["hgbr_rank_v2"][i]),
            "hgbr_q10_v2": float(comp["hgbr_q10_v2"][i]),
            "hgbr_q50_v2": float(comp["hgbr_q50_v2"][i]),
            "hgbr_q90_v2": float(comp["hgbr_q90_v2"][i]),
            "ensemble_v2": float(comp["score_final"][i]),
        }
        meta = {
            "score_final": float(comp["score_final"][i]),
            "mu": float(comp["mu"][i]),
            "uncert": float(comp["uncert"][i]),
            "score_rank_z": float(comp["score_rank_z"][i]),
        }
        for model_id, yhat in per_model.items():
            old = session.exec(
                select(MlPrediction)
                .where(MlPrediction.model_id == model_id)
                .where(MlPrediction.symbol == str(r["symbol"]))
                .where(MlPrediction.as_of_date == r["as_of_date"])
            ).first()
            m = meta if model_id == "ensemble_v2" else {"feature_version": "v2"}
            if old:
                old.y_hat = yhat
                old.meta = m
                session.add(old)
            else:
                session.add(
                    MlPrediction(
                        model_id=model_id,
                        symbol=str(r["symbol"]),
                        as_of_date=r["as_of_date"],
                        y_hat=yhat,
                        meta=m,
                    )
                )
        up += 1
    session.commit()
    return up


def run_diagnostics_v2(session: Session) -> int:
    from core.db.models import DiagnosticsMetric, DiagnosticsRun, ForeignRoom, MlPrediction, QuoteL2
    from core.ml.diagnostics import run_diagnostics
    from core.ml.features import build_ml_features
    from core.ml.features_v2 import build_features_v2

    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    preds = session.exec(select(MlPrediction).where(MlPrediction.model_id == "ensemble_v2")).all()
    if not prices or not preds:
        return 0
    f = build_ml_features(pd.DataFrame([p.model_dump() for p in prices]))
    fr = pd.DataFrame([r.model_dump() for r in session.exec(select(ForeignRoom)).all()])
    if not fr.empty:
        fr["as_of_date"] = pd.to_datetime(fr["timestamp"]).dt.date
        fr = fr.groupby(["symbol", "as_of_date"], as_index=False).last()
        fr["net_foreign_val"] = fr["fbuy_val"].fillna(0.0) - fr["fsell_val"].fillna(0.0)
    quotes = pd.DataFrame([r.model_dump() for r in session.exec(select(QuoteL2)).all()])
    intraday = pd.DataFrame(
        [
            r.model_dump()
            for r in session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1m")).all()
        ]
    )
    f = build_features_v2(
        f,
        None,
        fr if not fr.empty else None,
        quotes if not quotes.empty else None,
        intraday if not intraday.empty else None,
    )
    p = pd.DataFrame([x.model_dump() for x in preds])
    p["score_final"] = p["meta"].apply(
        lambda m: float((m or {}).get("score_final", 0.0)) if isinstance(m, dict) else 0.0
    )
    d = f.merge(
        p[["symbol", "as_of_date", "score_final"]], on=["symbol", "as_of_date"], how="inner"
    )
    d["net_ret"] = d["y_excess"].fillna(0.0)
    d["order_notional"] = 10_000_000.0
    d["turnover"] = 0.0
    d["commission"] = 0.0
    d["sell_tax"] = 0.0
    d["slippage_cost"] = 0.0
    d["liq_bound"] = False
    d["regime"] = np.where(
        d.get("regime_risk_off", 0.0) > 0.5,
        "risk_off",
        np.where(d.get("regime_trend_up", 0.0) > 0.5, "trend_up", "sideways"),
    )
    metrics = run_diagnostics(d)
    run_id = f"diag-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    session.add(DiagnosticsRun(run_id=run_id, model_id="ensemble_v2", config_hash="worker"))
    for k, v in metrics.items():
        session.add(DiagnosticsMetric(run_id=run_id, metric_name=k, metric_value=float(v)))
    session.commit()
    return len(metrics)


def ensure_partitions_monthly(session: Session) -> int:
    """Ensure +3 monthly partitions for high-volume postgres tables."""
    return ensure_partitions_monthly_impl(session, months_ahead=3)
