from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "bar_builder"))

from bar_builder.main import BarBuilderService
from bar_builder.storage import BarStorage
from tests.helpers_redis_fake import FakeRedisCompat


def test_late_event_generates_correction_and_no_mutation() -> None:
    redis = FakeRedisCompat()
    # first event in later interval then out-of-order older event becomes late (>10s)
    redis.xadd(
        "stream:market_events",
        {
            "payload": json.dumps(
                {
                    "event_type": "TRADE",
                    "event_id": "x1",
                    "symbol": "AAA",
                    "provider_ts": "2025-01-02T03:01:00Z",
                    "price": 12.0,
                    "qty": 50,
                    "payload_hash": "h1",
                }
            )
        },
    )
    redis.xadd(
        "stream:market_events",
        {
            "payload": json.dumps(
                {
                    "event_type": "TRADE",
                    "event_id": "x2",
                    "symbol": "AAA",
                    "provider_ts": "2025-01-02T02:05:00Z",
                    "price": 10.0,
                    "qty": 100,
                    "payload_hash": "h2",
                }
            )
        },
    )

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        svc = BarBuilderService(
            redis_client=redis, storage=BarStorage(session, redis), exchange="HOSE"
        )
        metrics = svc.run_once()

    assert metrics["late_events_total"] >= 1
    rows = redis.xrange("stream:market_events")
    assert any("CORRECTION" in str(v.get("payload", "")) for _, v in rows)
