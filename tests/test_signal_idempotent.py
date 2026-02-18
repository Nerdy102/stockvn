from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "realtime_signal_engine"))

from core.db.models import SignalIntraday
from realtime_signal_engine.engine import RealtimeSignalEngine
from realtime_signal_engine.storage import SignalStorage
from tests.helpers_redis_fake import FakeRedisCompat


def test_signal_idempotent() -> None:
    redis = FakeRedisCompat()
    payload = {
        "symbol": "AAA",
        "timeframe": "15m",
        "start_ts": "2025-01-02T02:00:00Z",
        "end_ts": "2025-01-02T02:15:00Z",
        "o": 10.0,
        "h": 10.0,
        "l": 10.0,
        "c": 10.0,
        "v": 100.0,
        "n_trades": 1,
        "vwap": 10.0,
    }
    redis.xadd("stream:bar_close:15m", {"payload": json.dumps(payload)})
    redis.xadd("stream:bar_close:15m", {"payload": json.dumps(payload)})

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        eng = RealtimeSignalEngine(
            redis_client=redis,
            storage=SignalStorage(s),
            config={"alert_expression": "close > EMA20"},
        )
        eng.run_once()
        eng.run_once()
        rows = s.exec(select(SignalIntraday)).all()
        assert len(rows) == 1
