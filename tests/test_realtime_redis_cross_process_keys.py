from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "bar_builder"))
sys.path.insert(0, str(ROOT / "services" / "realtime_signal_engine"))

from bar_builder.consumer import read_market_events
from bar_builder.storage import BarStorage
from realtime_signal_engine.state_store import StateStore


class RedisListKvStub:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.stream_rows: list[tuple[str, dict[str, str]]] = []
        self.cursor = "0-0"

    def get(self, key: str):
        return self.kv.get(key)

    def set(self, key: str, value: str):
        self.kv[key] = value
        return True

    def rpush(self, key: str, value: str):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def ltrim(self, key: str, start: int, end: int):
        arr = self.lists.get(key, [])
        n = len(arr)
        i = max(0, n + start if start < 0 else start)
        j = n + end if end < 0 else end
        j = min(n - 1, j)
        self.lists[key] = arr[i : j + 1] if j >= i else []
        return True

    def lrange(self, key: str, start: int, end: int):
        arr = self.lists.get(key, [])
        n = len(arr)
        i = max(0, n + start if start < 0 else start)
        j = n + end if end < 0 else end
        j = min(n - 1, j)
        return arr[i : j + 1] if j >= i else []

    def xread(self, streams: dict[str, str], block: int = 1000, count: int = 1000):
        del block, count
        _, last_id = next(iter(streams.items()))
        last_seq = int(str(last_id).split("-", 1)[0])
        rows = [r for r in self.stream_rows if int(r[0].split("-", 1)[0]) > last_seq]
        if not rows:
            return []
        return [("stream:market_events", rows)]

    def xrange(self, stream: str):
        del stream
        return list(self.stream_rows)


def test_bar_storage_cache_uses_redis_list_keys() -> None:
    redis = RedisListKvStub()
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        storage = BarStorage(session, redis)
        for i in range(220):
            storage.cache_bar(
                {
                    "symbol": "BTCUSDT",
                    "timeframe": "15m",
                    "start_ts": dt.datetime.utcnow().isoformat() + "Z",
                    "end_ts": dt.datetime.utcnow().isoformat() + "Z",
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
            )

    key = "realtime:bars:BTCUSDT:15m"
    assert len(redis.lists[key]) == 200
    assert json.loads(redis.lists[key][-1])["bar_hash"] == "219"


def test_state_store_uses_redis_kv_and_hot_list() -> None:
    redis = RedisListKvStub()
    store = StateStore(redis)

    payload = {"symbol": "BTCUSDT", "timeframe": "15m", "signal_hash": "h"}
    store.set_signal_snapshot("BTCUSDT", "15m", payload)
    store.set_ops_summary({"last_update": "2026-01-01T00:00:00Z"})
    store.push_hot("top_movers", {"symbol": "BTCUSDT"}, limit=2)

    assert "realtime:signals:BTCUSDT:15m" in redis.kv
    assert "realtime:ops:summary" in redis.kv
    assert len(redis.lists["realtime:hot:top_movers"]) == 1


def test_market_event_reader_tracks_cursor() -> None:
    redis = RedisListKvStub()
    redis.stream_rows = [
        ("1-0", {"payload": json.dumps({"event_id": "a", "event_type": "TRADE"})}),
        ("2-0", {"payload": json.dumps({"event_id": "b", "event_type": "TRADE"})}),
    ]

    first = read_market_events(redis)
    second = read_market_events(redis)

    assert [x["event_id"] for x in first] == ["a", "b"]
    assert second == []
    assert redis.kv["cursor:market_events"] == "2-0"
