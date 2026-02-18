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
    AlphaPrediction,
    AlertEvent,
    AlertRule,
    BronzeFile,
    DataQualityMetric,
    DriftAlert,
    DriftMetric,
    FactorScore,
    Fundamental,
    IndicatorState,
    IndicatorValue,
    JobRun,
    MlFeature,
    MlLabel,
    DsrResult,
    GateResult,
    MinTrlResult,
    PboResult,
    ParquetManifest,
    PsrResult,
    PriceOHLCV,
    QuoteL2,
    Signal,
    Ticker,
    DailyFlowFeature,
    DailyIntradayFeature,
    DailyOrderbookFeature,
    CorporateAction,
    FeatureCoverage,
    FeatureLastProcessed,
    RealityCheckResult,
    SpaResult,
)
from core.alpha_v3.bootstrap import block_bootstrap_ci as alpha_v3_block_bootstrap_ci
from core.corporate_actions import adjust_prices
from core.alpha_v3.dsr import compute_deflated_sharpe_ratio
from core.alpha_v3.gates import evaluate_research_gates
from core.alpha_v3.pbo import compute_pbo_cscv
from core.calendar_vn import get_trading_calendar_vn
from core.ml.backtest import run_sensitivity
from core.data_lake.feature_source import load_table_df
from core.data_lake.parquet_export import export_partitioned_parquet_for_day
from research.stats.psr_mintrl import compute_mintrl, compute_psr
from research.stats.reality_check import white_reality_check
from research.stats.spa_test import hansen_spa_test
from core.db.partitioning import ensure_partitions_monthly as ensure_partitions_monthly_impl
from core.features.daily_flow import compute_daily_flow_features as compute_daily_flow_features_impl
from core.features.daily_intraday import (
    compute_daily_intraday_features as compute_daily_intraday_features_impl,
)
from core.features.daily_orderbook import (
    compute_daily_orderbook_features as compute_daily_orderbook_features_impl,
)
from core.factors import compute_factors
from core.indicators import RSIState, add_indicators, ema_incremental, rsi_incremental
from core.logging import get_logger
from core.monitoring.data_quality import compute_data_quality_metrics
from core.monitoring.drift import compute_weekly_drift_metrics
from core.monitoring.prometheus_metrics import (
    INGEST_ERRORS_TOTAL,
    INGEST_ROWS_TOTAL,
    LAST_FEATURE_TS,
    LAST_INGEST_TS,
    LAST_TRAIN_TS,
    REDIS_STREAM_LAG,
    UPSERT_ROWS_TOTAL,
    mark_now,
)
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


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ml_feature_payload_from_row(row: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    allowed = set(MlFeature.model_fields.keys())
    payload: dict[str, Any] = {}
    for c in feature_cols:
        if c in row.index and c in allowed:
            payload[c] = _json_sanitize(row[c])
    return payload


def _coverage_metrics_for_date(df: pd.DataFrame, feature_cols: list[str]) -> tuple[float, dict[str, Any]]:
    if df.empty:
        metrics = {
            "missing_rate_per_feature": {c: 1.0 for c in feature_cols},
            "symbols_dropped_pct": 1.0,
            "critical_features": ["ret_21d", "vol_20d", "adv20_value", "rsi14"],
        }
        return 0.0, metrics

    miss = {c: float(df[c].isna().mean()) if c in df.columns else 1.0 for c in feature_cols}
    critical = [c for c in ["ret_21d", "vol_20d", "adv20_value", "rsi14"] if c in feature_cols]
    if critical:
        dropped = float(df[critical].isna().any(axis=1).mean())
    else:
        dropped = 0.0
    coverage = float(max(0.0, min(1.0, 1.0 - (sum(miss.values()) / max(1, len(miss))))))
    metrics = {
        "missing_rate_per_feature": miss,
        "symbols_dropped_pct": dropped,
        "critical_features": critical,
    }
    return coverage, metrics


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

    # Intraday seed range anchors strictly to last daily date from provider.
    last_date = _provider_last_date(provider, symbols)
    if last_date is None:
        raise RuntimeError("Cannot seed intraday data: provider has no daily anchor date.")
    start = get_trading_calendar_vn().shift_trading_days(last_date, -20)

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
    total_rows = 0
    for sym in tickers["symbol"].tolist():
        try:
            df = provider.get_ohlcv(sym, "1D")
            if not df.empty:
                ingest_prices(session, sym, "1D", df)
                rows = len(df)
                total_rows += rows
                INGEST_ROWS_TOTAL.labels(channel="prices_1d").inc(rows)
                UPSERT_ROWS_TOTAL.labels(table="price_ohlcv").inc(rows)
        except Exception:
            INGEST_ERRORS_TOTAL.labels(type="prices_job").inc()
            raise
    mark_now(LAST_INGEST_TS)
    update_market_caps(session)


def update_market_caps(session: Session) -> None:
    """Update tickers.market_cap based on latest 1D close (idempotent)."""
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return
    pdf = pd.DataFrame([p.model_dump() for p in prices])
    pdf = _adjust_daily_prices_with_ca(session, pdf)
    pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date
    last_close = pdf.sort_values(["symbol", "date"]).groupby("symbol")["close"].last()

    tickers = session.exec(select(Ticker)).all()
    for t in tickers:
        if t.symbol in last_close.index and t.shares_outstanding > 0:
            t.market_cap = float(last_close.loc[t.symbol]) * float(t.shares_outstanding)
            session.merge(t)
    session.commit()


def _adjust_daily_prices_with_ca(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Apply CA adjusted=True,total_return=False to daily bars before indicators/factors."""
    if df.empty:
        return df
    ca_rows = session.exec(select(CorporateAction)).all()
    if not ca_rows:
        return df
    ca_df = pd.DataFrame([c.model_dump() for c in ca_rows])
    out_parts: list[pd.DataFrame] = []
    for sym, g in df.groupby("symbol", sort=False):
        adj = adjust_prices(
            symbol=str(sym),
            bars=g.copy(),
            start=pd.to_datetime(g["timestamp"]).dt.date.min(),
            end=pd.to_datetime(g["timestamp"]).dt.date.max(),
            method="ca",
            corporate_actions=ca_df,
            total_return=False,
        )
        # keep source timestamp for downstream jobs
        if "timestamp" in g.columns and "timestamp" not in adj.columns:
            adj["timestamp"] = g["timestamp"].values
        out_parts.append(adj)
    out = pd.concat(out_parts, ignore_index=True)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def compute_indicators(session: Session) -> None:
    """Job: compute_indicators -> store IndicatorValue (idempotent)."""
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"])
    df = _adjust_daily_prices_with_ca(session, df)

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
    pdf = _adjust_daily_prices_with_ca(session, pdf)
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

    INGEST_ROWS_TOTAL.labels(channel="ssi_stream").inc(processed)
    UPSERT_ROWS_TOTAL.labels(table="stream_silver").inc(processed)
    try:
        REDIS_STREAM_LAG.set(float(sum(len(messages) for _, messages in rows) - acked))
    except Exception:
        pass
    if processed > 0:
        mark_now(LAST_INGEST_TS)

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
    UPSERT_ROWS_TOTAL.labels(table="data_quality_metrics").inc(len(metrics))
    mark_now(LAST_FEATURE_TS)
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
    spread_proxy = ((mkt["high"] - mkt["low"]) / mkt["close"].replace(0, pd.NA)).fillna(0.0).iloc[1:]
    flow_intensity = (mkt["value_vnd"].astype(float) / mkt["volume"].replace(0, pd.NA)).fillna(0.0).iloc[1:]
    metrics = compute_weekly_drift_metrics(ret, vol, spread_proxy, flow_intensity)

    today = dt.date.today()
    for m in metrics:
        metric_name = str(m.get("metric_name"))
        metric_value = float(m.get("metric_value", 0.0))
        should_alert = bool(m.get("alert", False))
        session.add(
            DriftMetric(
                metric_date=today,
                metric_name=metric_name,
                metric_value=metric_value,
                alert=should_alert,
            )
        )
        if should_alert:
            session.add(
                DriftAlert(
                    metric_date=today,
                    metric_name=metric_name,
                    psi_value=metric_value,
                    threshold=0.25,
                    message=f"PSI drift alert for {metric_name}: {metric_value:.4f}",
                )
            )

    run.status = "completed"
    run.end_ts = dt.datetime.utcnow()
    run.rows_in = len(df)
    run.rows_out = len(metrics)
    session.add(run)
    session.commit()
    UPSERT_ROWS_TOTAL.labels(table="drift_metrics").inc(len(metrics))
    mark_now(LAST_FEATURE_TS)
    return metrics


def compute_daily_flow_features(session: Session) -> int:
    return compute_daily_flow_features_impl(session)


def compute_daily_orderbook_features(session: Session) -> int:
    return compute_daily_orderbook_features_impl(session)


def compute_daily_intraday_features(session: Session) -> int:
    return compute_daily_intraday_features_impl(session)


def compute_intraday_daily_features(session: Session) -> int:
    """Backward-compatible alias."""
    return compute_daily_intraday_features(session)


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
            obj.y_rank_z = payload["y_rank_z"]
            obj.y_excess = payload["y_excess"]
            session.add(obj)
        else:
            session.add(
                MlFeature(
                    symbol=str(r["symbol"]),
                    as_of_date=r["as_of_date"],
                    feature_version="v2",
                    y_rank_z=payload["y_rank_z"],
                    y_excess=payload["y_excess"],
                )
            )
        up += 1
    session.commit()
    return up




def job_build_labels_v3(session: Session) -> int:
    from core.alpha_v3.targets import build_labels_v3

    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    if not prices:
        return 0
    labels = build_labels_v3(pd.DataFrame([p.model_dump() for p in prices]))

    up = 0
    for _, r in labels.dropna(subset=["y_excess", "y_rank_z"]).iterrows():
        obj = session.exec(
            select(MlLabel)
            .where(MlLabel.symbol == str(r["symbol"]))
            .where(MlLabel.date == r["date"])
            .where(MlLabel.label_version == "v3")
        ).first()
        if obj:
            obj.y_excess = float(r["y_excess"])
            obj.y_rank_z = float(r["y_rank_z"])
            session.add(obj)
        else:
            session.add(
                MlLabel(
                    symbol=str(r["symbol"]),
                    date=r["date"],
                    y_excess=float(r["y_excess"]),
                    y_rank_z=float(r["y_rank_z"]),
                    label_version="v3",
                )
            )
        up += 1
    session.commit()
    return up


def job_build_ml_features_v3(session: Session) -> int:
    from core.alpha_v3.features import build_ml_features_v3, feature_columns_v3

    session.commit()
    feature_version = "v3"
    settings = Settings()
    price_df = load_table_df(
        session=session,
        model=PriceOHLCV,
        table_name="prices_ohlcv",
        date_col="timestamp",
        settings=settings,
    )
    if price_df.empty:
        return 0
    if "timeframe" in price_df.columns:
        price_df = price_df[price_df["timeframe"] == "1D"].copy()
    price_df["date"] = pd.to_datetime(price_df["timestamp"]).dt.date

    state = session.exec(
        select(FeatureLastProcessed)
        .where(FeatureLastProcessed.feature_name == "ml_features_v3")
        .where(FeatureLastProcessed.symbol == feature_version)
    ).first()
    last_processed_date = state.last_date if state else None
    if last_processed_date is not None:
        pending_dates = sorted(d for d in price_df["date"].dropna().unique().tolist() if d > last_processed_date)
    else:
        pending_dates = sorted(price_df["date"].dropna().unique().tolist())
    if not pending_dates:
        return 0

    factors = pd.DataFrame([r.model_dump() for r in session.exec(select(FactorScore)).all()])
    fundamentals = pd.DataFrame([r.model_dump() for r in session.exec(select(Fundamental)).all()])
    tickers = pd.DataFrame([r.model_dump() for r in session.exec(select(Ticker)).all()])

    start_date = min(pending_dates) if pending_dates else None
    flow = load_table_df(session, DailyFlowFeature, "daily_flow_features", "date", settings, start_date=start_date)
    orderbook = load_table_df(session, DailyOrderbookFeature, "daily_orderbook_features", "date", settings, start_date=start_date)
    intra = load_table_df(session, DailyIntradayFeature, "daily_intraday_features", "date", settings, start_date=start_date)

    feat = build_ml_features_v3(
        prices=price_df,
        factors=factors if not factors.empty else None,
        fundamentals=fundamentals if not fundamentals.empty else None,
        flow_features=flow if not flow.empty else None,
        orderbook_features=orderbook if not orderbook.empty else None,
        intraday_features=intra if not intra.empty else None,
        tickers=tickers if not tickers.empty else None,
    )

    feat = feat[feat["date"].isin(pending_dates)].copy()
    if feat.empty:
        return 0

    feat_cols = feature_columns_v3(feat)
    all_symbols = sorted(feat["symbol"].dropna().astype(str).unique().tolist())
    max_date = max(pending_dates)

    up = 0
    for symbol_batch in _chunked(all_symbols, 200):
        batch_df = feat[feat["symbol"].astype(str).isin(symbol_batch)].copy()
        for date_batch in _chunked(pending_dates, 30):
            slice_df = batch_df[batch_df["date"].isin(date_batch)].copy()
            if slice_df.empty:
                continue

            for day, day_df in slice_df.groupby("date", sort=True):
                coverage_score, coverage_metrics = _coverage_metrics_for_date(day_df, feat_cols)
                cov = session.exec(
                    select(FeatureCoverage)
                    .where(FeatureCoverage.date == day)
                    .where(FeatureCoverage.feature_version == feature_version)
                ).first()
                if cov is None:
                    cov = FeatureCoverage(date=day, feature_version=feature_version, metrics_json=coverage_metrics)
                else:
                    cov.metrics_json = coverage_metrics
                session.add(cov)

                for _, r in day_df.iterrows():
                    payload = _ml_feature_payload_from_row(r, feat_cols)
                    obj = session.exec(
                        select(MlFeature)
                        .where(MlFeature.symbol == str(r["symbol"]))
                        .where(MlFeature.as_of_date == r["date"])
                        .where(MlFeature.feature_version == feature_version)
                    ).first()
                    if obj:
                        for k, v in payload.items():
                            setattr(obj, k, v)
                        obj.data_coverage_score = coverage_score
                        session.add(obj)
                    else:
                        session.add(
                            MlFeature(
                                symbol=str(r["symbol"]),
                                as_of_date=r["date"],
                                feature_version=feature_version,
                                data_coverage_score=coverage_score,
                                **payload,
                            )
                        )
                    up += 1
            session.commit()

    if state is None:
        state = FeatureLastProcessed(
            feature_name="ml_features_v3",
            symbol=feature_version,
            last_date=max_date,
            updated_at=dt.datetime.utcnow(),
        )
    else:
        state.last_date = max_date
        state.updated_at = dt.datetime.utcnow()
    session.add(state)
    session.commit()
    return up

def train_models_v2(session: Session) -> int:
    from core.alpha_v3.features import enforce_no_leakage_guard
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
    leakage_check = feat[["as_of_date"]].drop_duplicates().rename(columns={"as_of_date": "date"})
    cal = get_trading_calendar_vn()
    leakage_check["label_date"] = leakage_check["date"].map(
        lambda d: cal.shift_trading_days(pd.to_datetime(d).date(), 21)
    )
    enforce_no_leakage_guard(leakage_check, horizon=21)
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



def _git_commit_hash() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _alpha_v3_training_frame(session: Session) -> pd.DataFrame:
    feat_rows = session.exec(select(MlFeature).where(MlFeature.feature_version == "v3")).all()
    label_rows = session.exec(select(MlLabel).where(MlLabel.label_version == "v3")).all()
    if not feat_rows or not label_rows:
        return pd.DataFrame()

    feat_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.model_dump().items()
                if k not in {"id", "created_at"}
            }
            for r in feat_rows
        ]
    )
    feat_df = feat_df.drop(columns=["y_rank_z", "y_excess"], errors="ignore")
    lbl_df = pd.DataFrame(
        [{"symbol": r.symbol, "date": r.date, "y_rank_z": r.y_rank_z} for r in label_rows]
    )
    merged = feat_df.merge(
        lbl_df,
        left_on=["symbol", "as_of_date"],
        right_on=["symbol", "date"],
        how="inner",
    )
    merged = merged.dropna(subset=["y_rank_z"]).reset_index(drop=True)
    return merged


def train_alpha_v3(
    session: Session,
    artifact_root: str = "artifacts/models/alpha_v3",
) -> dict[str, Any] | None:
    from core.alpha_v3.models import (
        AlphaV3Config,
        AlphaV3ModelBundle,
        feature_matrix_from_records,
        metadata_payload,
        write_metadata,
    )
    from core.db.models import AlphaModel

    data = _alpha_v3_training_frame(session)
    if data.empty:
        return None

    config = AlphaV3Config()
    x, cols = feature_matrix_from_records(data)
    y = data["y_rank_z"].astype(float)

    bundle = AlphaV3ModelBundle(config=config).fit(x, y)
    version_dt = dt.datetime.utcnow().replace(microsecond=0)
    while True:
        version = version_dt.strftime("%Y%m%d_%H%M%S")
        artifact_dir = Path(artifact_root) / version
        if not artifact_dir.exists():
            break
        version_dt = version_dt + dt.timedelta(seconds=1)
    bundle.dump(artifact_dir)

    train_start = data["as_of_date"].min()
    train_end = data["as_of_date"].max()
    meta = metadata_payload(
        train_start,
        train_end,
        _git_commit_hash(),
        config,
        feature_columns=cols,
    )
    write_metadata(artifact_dir / "metadata.json", meta)

    config_hash = config.config_hash()
    model_row = session.exec(
        select(AlphaModel)
        .where(AlphaModel.model_id == config.model_id)
        .where(AlphaModel.version == version)
    ).first()
    if model_row is None:
        model_row = AlphaModel(
            model_id=config.model_id,
            version=version,
            artifact_path=str(artifact_dir),
            train_start=train_start,
            train_end=train_end,
            config_hash=config_hash,
        )
    else:
        model_row.artifact_path = str(artifact_dir)
        model_row.train_start = train_start
        model_row.train_end = train_end
        model_row.config_hash = config_hash
    session.add(model_row)
    session.commit()

    return {
        "model_id": config.model_id,
        "version": version,
        "artifact_path": str(artifact_dir),
        "train_start": train_start,
        "train_end": train_end,
        "config_hash": config_hash,
        "feature_cols": cols,
    }


def predict_alpha_v3(session: Session, as_of_date: dt.date, version: str | None = None) -> int:
    from core.alpha_v3.models import (
        align_feature_matrix,
        feature_matrix_from_records,
        load_alpha_v3_bundle,
        read_metadata,
    )
    from core.db.models import AlphaModel, AlphaPrediction

    q = select(AlphaModel).where(AlphaModel.model_id == "alpha_v3")
    if version:
        q = q.where(AlphaModel.version == version)
    model_row = session.exec(q.order_by(AlphaModel.version.desc())).first()
    if model_row is None:
        return 0

    artifact_dir = Path(model_row.artifact_path)
    bundle = load_alpha_v3_bundle(artifact_dir)
    metadata = read_metadata(artifact_dir / "metadata.json")
    train_feature_cols = metadata.get("feature_columns")

    feat_rows = session.exec(
        select(MlFeature)
        .where(MlFeature.feature_version == "v3")
        .where(MlFeature.as_of_date == as_of_date)
    ).all()
    if not feat_rows:
        return 0

    feat_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.model_dump().items()
                if k not in {"id", "created_at"}
            }
            for r in feat_rows
        ]
    )
    x_raw, inferred_cols = feature_matrix_from_records(feat_df)
    if isinstance(train_feature_cols, list) and train_feature_cols:
        x = align_feature_matrix(x_raw, [str(c) for c in train_feature_cols])
    else:
        x = align_feature_matrix(x_raw, inferred_cols)
    comp = bundle.predict_components(x)

    symbols = [str(x) for x in feat_df["symbol"].tolist()]
    existing_rows = session.exec(
        select(AlphaPrediction)
        .where(AlphaPrediction.model_id == "alpha_v3")
        .where(AlphaPrediction.as_of_date == as_of_date)
    ).all()
    existing = {r.symbol: r for r in existing_rows}

    now = dt.datetime.utcnow()
    for i, symbol in enumerate(symbols):
        row = existing.get(symbol)
        if row is None:
            row = AlphaPrediction(
                model_id="alpha_v3",
                as_of_date=as_of_date,
                symbol=symbol,
                score=float(comp["score"][i]),
                mu=float(comp["mu"][i]),
                uncert=float(comp["uncert"][i]),
                pred_base=float(comp["pred_base"][i]),
                created_at=now,
            )
        else:
            row.score = float(comp["score"][i])
            row.mu = float(comp["mu"][i])
            row.uncert = float(comp["uncert"][i])
            row.pred_base = float(comp["pred_base"][i])
        session.add(row)

    session.commit()
    return len(symbols)


def job_train_alpha_v3(session: Session, as_of_date: dt.date | None = None) -> dict[str, int]:
    trained = train_alpha_v3(session)
    if trained is None:
        return {"trained": 0, "predictions": 0}

    pred_date = as_of_date
    if pred_date is None:
        pred_date = session.exec(
            select(MlFeature.as_of_date)
            .where(MlFeature.feature_version == "v3")
            .order_by(MlFeature.as_of_date.desc())
        ).first()
    if pred_date is None:
        return {"trained": 1, "predictions": 0}

    preds = predict_alpha_v3(session, as_of_date=pred_date, version=trained["version"])
    cp = job_update_alpha_v3_cp(session, as_of_date=pred_date)
    mark_now(LAST_TRAIN_TS)
    return {"trained": 1, "predictions": preds + int(cp.get("predictions", 0))}




def train_alpha_rankpair_v1(
    session: Session,
    artifact_root: str = "artifacts/models/alpha_rankpair_v1",
) -> dict[str, Any] | None:
    import pickle

    from core.alpha_v3.models import feature_matrix_from_records
    from core.db.models import AlphaModel
    from core.ml.rank_pairwise import PairwiseRankConfig, train_pairwise_ranker

    data = _alpha_v3_training_frame(session)
    if data.empty:
        return None

    x_frame, cols = feature_matrix_from_records(data)
    train_frame = pd.concat([data[["symbol", "as_of_date", "y_rank_z"]], x_frame], axis=1)

    cfg = PairwiseRankConfig()
    model = train_pairwise_ranker(train_frame, feature_columns=cols, config=cfg)
    if model is None:
        return None

    version_dt = dt.datetime.utcnow().replace(microsecond=0)
    while True:
        version = version_dt.strftime("%Y%m%d_%H%M%S")
        artifact_dir = Path(artifact_root) / version
        if not artifact_dir.exists():
            break
        version_dt = version_dt + dt.timedelta(seconds=1)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "feature_columns": model.feature_columns,
        "config": model.config,
        "estimator": model.estimator,
    }
    with (artifact_dir / "model.pkl").open("wb") as f:
        pickle.dump(payload, f)

    train_start = data["as_of_date"].min()
    train_end = data["as_of_date"].max()
    config_hash = f"rankpair-{len(cols)}-{model.config.pairs_per_date}-{model.config.cv_splits}"

    model_row = session.exec(
        select(AlphaModel)
        .where(AlphaModel.model_id == model.config.model_id)
        .where(AlphaModel.version == version)
    ).first()
    if model_row is None:
        model_row = AlphaModel(
            model_id=model.config.model_id,
            version=version,
            artifact_path=str(artifact_dir),
            train_start=train_start,
            train_end=train_end,
            config_hash=config_hash,
        )
    else:
        model_row.artifact_path = str(artifact_dir)
        model_row.train_start = train_start
        model_row.train_end = train_end
        model_row.config_hash = config_hash
    session.add(model_row)
    session.commit()

    return {
        "model_id": model.config.model_id,
        "version": version,
        "artifact_path": str(artifact_dir),
        "feature_cols": cols,
    }


def predict_alpha_rankpair_v1(session: Session, as_of_date: dt.date, version: str | None = None) -> int:
    import pickle

    from core.db.models import AlphaModel
    from core.ml.rank_pairwise import PairwiseRankModel, predict_rank_score

    q = select(AlphaModel).where(AlphaModel.model_id == "alpha_rankpair_v1")
    if version:
        q = q.where(AlphaModel.version == version)
    model_row = session.exec(q.order_by(AlphaModel.version.desc())).first()
    if model_row is None:
        return 0

    artifact_dir = Path(model_row.artifact_path)
    model_file = artifact_dir / "model.pkl"
    if not model_file.exists():
        return 0

    with model_file.open("rb") as f:
        payload = pickle.load(f)
    model = PairwiseRankModel(
        config=payload["config"],
        feature_columns=[str(c) for c in payload["feature_columns"]],
        estimator=payload["estimator"],
    )

    feat_rows = session.exec(
        select(MlFeature)
        .where(MlFeature.feature_version == "v3")
        .where(MlFeature.as_of_date == as_of_date)
    ).all()
    if not feat_rows:
        return 0

    feat_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.model_dump().items()
                if k not in {"id", "created_at"}
            }
            for r in feat_rows
        ]
    )
    pred_df = predict_rank_score(model, feat_df, date_col="as_of_date")
    pred_df = pred_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    existing_rows = session.exec(
        select(AlphaPrediction)
        .where(AlphaPrediction.model_id == "alpha_rankpair_v1")
        .where(AlphaPrediction.as_of_date == as_of_date)
    ).all()
    existing = {r.symbol: r for r in existing_rows}

    now = dt.datetime.utcnow()
    for _, r in pred_df.iterrows():
        symbol = str(r["symbol"])
        row = existing.get(symbol)
        score = float(r["score_z"])
        raw_score = float(r["raw_score"])
        if row is None:
            row = AlphaPrediction(
                model_id="alpha_rankpair_v1",
                as_of_date=as_of_date,
                symbol=symbol,
                score=score,
                mu=raw_score,
                uncert=0.0,
                pred_base=raw_score,
                created_at=now,
            )
        else:
            row.score = score
            row.mu = raw_score
            row.uncert = 0.0
            row.pred_base = raw_score
        session.add(row)

    session.commit()
    return len(pred_df)


def job_train_alpha_rankpair_v1(session: Session, as_of_date: dt.date | None = None) -> dict[str, int]:
    trained = train_alpha_rankpair_v1(session)
    if trained is None:
        return {"trained": 0, "predictions": 0}

    pred_date = as_of_date
    if pred_date is None:
        pred_date = session.exec(
            select(MlFeature.as_of_date)
            .where(MlFeature.feature_version == "v3")
            .order_by(MlFeature.as_of_date.desc())
        ).first()
    if pred_date is None:
        return {"trained": 1, "predictions": 0}

    preds = predict_alpha_rankpair_v1(session, as_of_date=pred_date, version=trained["version"])
    return {"trained": 1, "predictions": preds}




def job_update_alpha_v3_cp(session: Session, as_of_date: dt.date | None = None) -> dict[str, int]:
    from core.alpha_v3.conformal import apply_cp_predictions, update_delayed_residuals

    pred_date = as_of_date
    if pred_date is None:
        pred_date = session.exec(
            select(AlphaPrediction.as_of_date)
            .where(AlphaPrediction.model_id == "alpha_v3")
            .order_by(AlphaPrediction.as_of_date.desc())
        ).first()
    if pred_date is None:
        return {"updated_residuals": 0, "predictions": 0}

    updated_residuals = update_delayed_residuals(session, pred_date)
    preds = apply_cp_predictions(session, pred_date)
    return {"updated_residuals": updated_residuals, "predictions": preds}


def nightly_parquet_export_job(session: Session, as_of_date: dt.date | None = None) -> dict[str, Any]:
    target_day = as_of_date or (dt.datetime.utcnow().date() - dt.timedelta(days=1))
    settings = Settings()
    counts = export_partitioned_parquet_for_day(session, settings=settings, as_of_date=target_day)
    manifests = session.exec(
        select(ParquetManifest).where(ParquetManifest.year == target_day.year).where(ParquetManifest.month == target_day.month).where(ParquetManifest.day == target_day.day)
    ).all()
    return {
        "status": "ok",
        "as_of_date": target_day.isoformat(),
        "datasets": counts,
        "manifest_rows": len(manifests),
    }




def run_overfit_controls_alpha_v3(session: Session, run_id: str) -> dict[str, Any]:
    """Persist DSR/PBO/Bootstrap controls from available alpha_v3 prediction-label panel."""
    preds = pd.DataFrame(
        [
            p.model_dump()
            for p in session.exec(select(AlphaPrediction).where(AlphaPrediction.model_id == "alpha_v3")).all()
        ]
    )
    labels = pd.DataFrame(
        [
            {"symbol": l.symbol, "as_of_date": l.date, "y_rank_z": l.y_rank_z}
            for l in session.exec(select(MlLabel).where(MlLabel.label_version == "v3")).all()
        ]
    )
    if preds.empty or labels.empty:
        return {"status": "insufficient_data"}

    merged = preds.merge(labels, on=["symbol", "as_of_date"], how="inner")
    if merged.empty:
        return {"status": "insufficient_data"}

    daily = (
        merged.groupby("as_of_date", as_index=False)
        .apply(lambda g: pd.Series({"net_ret": float(g.nlargest(5, "score")["y_rank_z"].mean())}))
        .reset_index(drop=True)
    )
    returns = pd.Series(daily["net_ret"], dtype=float)

    sensitivity = run_sensitivity(base_metrics={}, base_returns=returns)
    variants_df = pd.DataFrame(
        {k: pd.Series(v, dtype=float) for k, v in (sensitivity.get("variant_returns", {}) or {}).items()}
    )
    if variants_df.empty:
        variants_df = pd.DataFrame({"base": returns, "base_cost": returns - 0.0001})

    n_trials = len(variants_df.columns)
    dsr = compute_deflated_sharpe_ratio(returns, n_trials=n_trials)
    phi, logits_summary = compute_pbo_cscv(variants_df, slices=10)
    psr = compute_psr(returns, sr_threshold=0.0)
    mintrl = compute_mintrl(returns, sr_threshold=0.0, alpha=0.95)
    rc_p, rc_components = white_reality_check(returns, variants_df, n_bootstrap=1000, block_mean=20.0, seed=42)
    spa_p, spa_components = hansen_spa_test(returns, variants_df, n_bootstrap=1000, block_mean=20.0, seed=42)
    ci = alpha_v3_block_bootstrap_ci(returns, block=20, n_resamples=1000)
    gates = evaluate_research_gates(
        dsr_value=dsr.dsr_value,
        pbo_phi=phi,
        psr_value=psr.psr_value,
        rc_p_value=rc_p,
        spa_p_value=spa_p,
    )

    dsr_row = session.exec(select(DsrResult).where(DsrResult.run_id == run_id)).first()
    if dsr_row is None:
        dsr_row = DsrResult(run_id=run_id, dsr_value=dsr.dsr_value, components=dsr.components)
    else:
        dsr_row.dsr_value = dsr.dsr_value
        dsr_row.components = dsr.components
    session.add(dsr_row)

    pbo_row = session.exec(select(PboResult).where(PboResult.run_id == run_id)).first()
    if pbo_row is None:
        pbo_row = PboResult(run_id=run_id, phi=phi, logits_summary=logits_summary)
    else:
        pbo_row.phi = phi
        pbo_row.logits_summary = logits_summary
    session.add(pbo_row)

    psr_row = session.exec(select(PsrResult).where(PsrResult.run_id == run_id)).first()
    if psr_row is None:
        psr_row = PsrResult(
            run_id=run_id,
            psr_value=psr.psr_value,
            sr_hat=psr.sr_hat,
            sr_threshold=psr.sr_threshold,
            t=psr.t,
            skew=psr.skew,
            kurt=psr.kurt,
        )
    else:
        psr_row.psr_value = psr.psr_value
        psr_row.sr_hat = psr.sr_hat
        psr_row.sr_threshold = psr.sr_threshold
        psr_row.t = psr.t
        psr_row.skew = psr.skew
        psr_row.kurt = psr.kurt
    session.add(psr_row)

    mintrl_row = session.exec(select(MinTrlResult).where(MinTrlResult.run_id == run_id)).first()
    mintrl_value = 0 if np.isinf(mintrl.mintrl) else int(mintrl.mintrl)
    if mintrl_row is None:
        mintrl_row = MinTrlResult(
            run_id=run_id,
            mintrl=mintrl_value,
            sr_hat=mintrl.sr_hat,
            sr_threshold=mintrl.sr_threshold,
            alpha=mintrl.alpha,
        )
    else:
        mintrl_row.mintrl = mintrl_value
        mintrl_row.sr_hat = mintrl.sr_hat
        mintrl_row.sr_threshold = mintrl.sr_threshold
        mintrl_row.alpha = mintrl.alpha
    session.add(mintrl_row)

    rc_row = session.exec(select(RealityCheckResult).where(RealityCheckResult.run_id == run_id)).first()
    if rc_row is None:
        rc_row = RealityCheckResult(run_id=run_id, p_value=rc_p, components=rc_components)
    else:
        rc_row.p_value = rc_p
        rc_row.components = rc_components
    session.add(rc_row)

    spa_row = session.exec(select(SpaResult).where(SpaResult.run_id == run_id)).first()
    if spa_row is None:
        spa_row = SpaResult(run_id=run_id, p_value=spa_p, components=spa_components)
    else:
        spa_row.p_value = spa_p
        spa_row.components = spa_components
    session.add(spa_row)

    gate_row = session.exec(select(GateResult).where(GateResult.run_id == run_id)).first()
    if gate_row is None:
        gate_row = GateResult(run_id=run_id, status=gates.get("status", "FAIL"), reasons={"reasons": gates.get("reasons", [])}, details=gates)
    else:
        gate_row.status = gates.get("status", "FAIL")
        gate_row.reasons = {"reasons": gates.get("reasons", [])}
        gate_row.details = gates
    session.add(gate_row)
    session.commit()

    return {
        "status": "ok",
        "run_id": run_id,
        "dsr": dsr.dsr_value,
        "pbo": phi,
        "psr": psr.psr_value,
        "mintrl": mintrl_value,
        "rc_p": rc_p,
        "spa_p": spa_p,
        "gate": gates,
        "ci": ci,
    }
