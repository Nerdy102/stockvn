from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.db.models import BronzeRaw, IngestState, PriceOHLCV
from core.metrics import METRICS
from core.settings import get_settings
from sqlmodel import Session, select

from data.adapters.ssi_mapper import SSIMapper
from data.bronze.writer import BronzeWriter
from data.schemas.ssi_fcdata import DailyOhlcRecord

log = logging.getLogger(__name__)


def payload_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append_bronze(
    session: Session,
    provider_name: str,
    endpoint_or_channel: str,
    payload: dict[str, Any],
    trading_date: dt.date | None = None,
    symbol: str | None = None,
    index_id: str | None = None,
) -> bool:
    p_hash = payload_hash(payload)
    exists = session.exec(
        select(BronzeRaw)
        .where(BronzeRaw.provider_name == provider_name)
        .where(BronzeRaw.endpoint_or_channel == endpoint_or_channel)
        .where(BronzeRaw.payload_hash == p_hash)
    ).first()
    if exists:
        return False
    raw = BronzeRaw(
        provider_name=provider_name,
        endpoint_or_channel=endpoint_or_channel,
        trading_date=trading_date,
        symbol=symbol,
        index_id=index_id,
        payload_hash=p_hash,
        raw_payload=json.dumps(payload, ensure_ascii=False, default=str),
    )
    session.add(raw)
    METRICS.inc("ingest_events_total", channel=endpoint_or_channel)
    return True


def upsert_silver_price(
    session: Session, payload: dict[str, Any], source: str = "ssi_fcdata"
) -> PriceOHLCV:
    bar = SSIMapper.map_daily_ohlc(payload)
    row = PriceOHLCV(
        symbol=bar.symbol,
        timeframe=bar.timeframe,
        timestamp=bar.ts,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        value_vnd=bar.value or 0.0,
        source=source,
        quality_flags={"flags": bar.quality_flags},
    )
    session.merge(row)
    METRICS.inc("db_upsert_rows_total", table="prices_ohlcv")
    return row


def update_ingest_state(
    session: Session,
    provider: str,
    channel: str,
    symbol: str,
    last_ts: dt.datetime | None,
    last_cursor: str | None = None,
) -> None:
    state = IngestState(
        provider=provider,
        channel=channel,
        symbol=symbol,
        last_ts=last_ts,
        last_cursor=last_cursor,
        updated_at=dt.datetime.utcnow(),
    )
    session.merge(state)


def ingest_from_fixtures(
    session: Session, fixture_dir: str = "tests/fixtures/ssi_fcdata"
) -> dict[str, int]:
    if should_pause_ingest():
        return {"bronze_added": 0, "silver_processed": 0}

    fdir = Path(fixture_dir)
    prices_payload = json.loads((fdir / "daily_ohlc.json").read_text())

    bronze_added = 0
    silver_rows = 0
    writer = BronzeWriter(provider="ssi_fcdata", channel="DailyOhlc", session=session)
    for payload in prices_payload:
        writer.write(payload)
    writer.flush()

    for rec in prices_payload:
        parsed = DailyOhlcRecord.model_validate(rec)
        added = append_bronze(
            session,
            provider_name="ssi_fcdata",
            endpoint_or_channel="DailyOhlc",
            payload=rec,
            trading_date=parsed.TradingDate,
            symbol=parsed.Symbol,
        )
        bronze_added += int(added)
        upsert_silver_price(session, rec)
        silver_rows += 1
        update_ingest_state(session, "ssi_fcdata", "DailyOhlc", parsed.Symbol, parsed.Time)

    session.commit()
    log.info(
        "ingest_from_fixtures_completed", extra={"event": "ingest_done", "provider": "ssi_fcdata"}
    )
    return {"bronze_added": bronze_added, "silver_processed": silver_rows}


def should_pause_ingest() -> bool:
    settings = get_settings()
    snap = METRICS.snapshot()
    err = sum(v for k, v in snap.items() if k.startswith("ingest_errors_total"))
    drift = sum(v for k, v in snap.items() if k.startswith("schema_unknown_fields_total"))
    return (
        err >= settings.INGEST_ERROR_KILL_SWITCH_THRESHOLD
        or drift >= settings.INGEST_SCHEMA_DRIFT_KILL_SWITCH_THRESHOLD
    )
