from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path

from sqlmodel import Session, select

from core.db.models import EventLog
from core.db.session import create_db_and_tables, get_engine

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


def _parse_ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def load_events(session: Session, start_ts: dt.datetime, end_ts: dt.datetime) -> list[EventLog]:
    stmt = (
        select(EventLog)
        .where(EventLog.ts_utc >= start_ts)
        .where(EventLog.ts_utc <= end_ts)
        .order_by(EventLog.ts_utc, EventLog.id)
    )
    return list(session.exec(stmt))


def _stream_key(event: EventLog) -> str:
    payload = event.payload_json if isinstance(event.payload_json, dict) else {}
    if "stream_key" in payload:
        return str(payload["stream_key"])
    if event.event_type in {"X", "X-QUOTE", "X-TRADE", "R", "MI", "B", "F", "OL"}:
        return f"ssi:{event.event_type}"
    return f"ssi:{event.event_type}"


def replay_into_redis(
    events: Iterable[EventLog],
    redis_client: object | None,
    speed: str = "1x",
    dry_run: bool = False,
) -> int:
    speed_factor = {"1x": 1.0, "10x": 10.0, "max": None}[speed]

    count = 0
    prev_event_ts: dt.datetime | None = None
    last_ts_by_symbol: dict[str, dt.datetime] = defaultdict(lambda: dt.datetime.min)

    for event in events:
        key = _stream_key(event)
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        symbol = event.symbol or payload.get("symbol", "")
        current_ts = event.ts_utc

        if symbol and current_ts < last_ts_by_symbol[str(symbol)]:
            raise AssertionError(f"ordering violation for symbol={symbol}")
        if symbol:
            last_ts_by_symbol[str(symbol)] = current_ts

        if not dry_run:
            if redis_client is None:
                raise RuntimeError("redis client is required when dry_run=false")
            redis_client.xadd(
                key,
                {
                    "event_id": str(event.id or ""),
                    "ts_utc": event.ts_utc.isoformat(),
                    "symbol": str(symbol),
                    "event_type": event.event_type,
                    "payload": json.dumps(payload, ensure_ascii=False, default=str),
                },
            )
        count += 1

        if speed_factor is None:
            prev_event_ts = current_ts
            continue
        if prev_event_ts is not None:
            delta_s = max((current_ts - prev_event_ts).total_seconds(), 0.0)
            time.sleep(delta_s / speed_factor)
        prev_event_ts = current_ts
    return count


def _load_fixture_events(path: Path) -> list[EventLog]:
    rows: list[EventLog] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(
            EventLog(
                id=idx,
                ts_utc=_parse_ts(str(obj["ts_utc"])),
                source=str(obj.get("source", "fixture")),
                event_type=str(obj["event_type"]),
                symbol=str(obj.get("symbol", "")),
                payload_json=obj.get("payload_json", {}),
                payload_hash=str(obj.get("payload_hash", "")),
            )
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Replay canonical event_log into Redis streams")
    p.add_argument("--start-ts")
    p.add_argument("--end-ts")
    p.add_argument("--fixture")
    p.add_argument("--speed", choices=["1x", "10x", "max"], default="1x")
    p.add_argument("--database-url", default=os.getenv("DATABASE_URL", "sqlite:///./vn_invest.db"))
    p.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if redis is None and not args.dry_run:
        raise RuntimeError("redis package is required for replay")

    if args.fixture:
        events = _load_fixture_events(Path(args.fixture))
    else:
        if not args.start_ts or not args.end_ts:
            raise ValueError("--start-ts and --end-ts are required when --fixture is not provided")
        create_db_and_tables(args.database_url)
        engine = get_engine(args.database_url)
        with Session(engine) as session:
            events = load_events(session, _parse_ts(args.start_ts), _parse_ts(args.end_ts))

    redis_client = None if args.dry_run else redis.from_url(args.redis_url, decode_responses=True)
    replayed = replay_into_redis(events, redis_client, speed=args.speed, dry_run=args.dry_run)
    print(json.dumps({"replayed": replayed}))


if __name__ == "__main__":
    main()
