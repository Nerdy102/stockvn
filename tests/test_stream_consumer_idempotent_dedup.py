from __future__ import annotations

from pathlib import Path

from core.db.models import BronzeRaw, PriceOHLCV, StreamDedup
from helpers_redis_fake import FakeRedisCompat
from sqlmodel import Session, SQLModel, create_engine, select
from worker_scheduler.jobs import consume_ssi_stream_to_bronze_silver

FIX = Path("tests/fixtures/ssi_streaming")


def _build_fake_redis():
    try:
        import fakeredis  # type: ignore

        return fakeredis.FakeRedis(decode_responses=True)
    except Exception:
        return FakeRedisCompat()


def test_stream_consumer_idempotent_dedup(monkeypatch) -> None:
    fake = _build_fake_redis()
    monkeypatch.setenv("REDIS_URL", "redis://unused")
    monkeypatch.setattr("worker_scheduler.jobs._create_redis_client", lambda _url: fake)

    payload = (FIX / "msg_B.json").read_text(encoding="utf-8")
    fake.xadd("ssi:B", {"ts_recv_utc": "1", "payload": payload, "rtype": "B"})
    fake.xadd("ssi:B", {"ts_recv_utc": "2", "payload": payload, "rtype": "B"})

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        first = consume_ssi_stream_to_bronze_silver(session)
        second = consume_ssi_stream_to_bronze_silver(session)

        assert first["processed"] == 1
        assert first["skipped"] == 1
        assert second["processed"] == 0

        dedups = session.exec(select(StreamDedup)).all()
        bronze = session.exec(select(BronzeRaw)).all()
        bars = session.exec(select(PriceOHLCV)).all()
        assert len(dedups) == 1
        assert len(bronze) == 1
        assert len(bars) == 1
