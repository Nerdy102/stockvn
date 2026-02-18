from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "bar_builder"))

from bar_builder.storage import BarStorage
from tests.helpers_redis_fake import FakeRedisCompat


def test_hot_cache_keeps_last_200() -> None:
    redis = FakeRedisCompat()
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        storage = BarStorage(session, redis)
        for i in range(250):
            row = {
                "symbol": "AAA",
                "timeframe": "15m",
                "start_ts": f"2025-01-02T02:{i%60:02d}:00+00:00",
                "end_ts": f"2025-01-02T02:{(i+1)%60:02d}:00+00:00",
                "o": 1,
                "h": 1,
                "l": 1,
                "c": 1,
                "v": 1,
                "n_trades": 1,
                "vwap": 1,
                "finalized": True,
                "bar_hash": str(i),
                "lineage_payload_hashes_json": [],
            }
            storage.cache_bar(row)

    arr = redis._bar_cache["realtime:bars:AAA:15m"]
    assert len(arr) == 200
    last = json.loads(arr[-1])
    assert last["bar_hash"] == "249"
