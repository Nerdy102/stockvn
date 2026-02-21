from __future__ import annotations

import argparse
import time

from core.db.session import create_db_and_tables, get_engine
from core.settings import get_settings
from redis import Redis
from sqlmodel import Session

from services.realtime_signal_engine.realtime_signal_engine.engine import RealtimeSignalEngine
from services.realtime_signal_engine.realtime_signal_engine.storage import SignalStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime signal engine loop")
    parser.add_argument("--redis", default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    redis_url = args.redis or settings.REDIS_URL
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    redis_client = Redis.from_url(redis_url, decode_responses=True)

    with Session(engine) as session:
        storage = SignalStorage(session)
        svc = RealtimeSignalEngine(redis_client=redis_client, storage=storage)
        while True:
            svc.run_once()
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
