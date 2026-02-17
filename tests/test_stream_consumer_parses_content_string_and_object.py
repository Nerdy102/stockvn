from __future__ import annotations

from pathlib import Path

from core.db.models import QuoteL2, TradeTape
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


def test_stream_consumer_parses_content_string_and_object(monkeypatch) -> None:
    fake = _build_fake_redis()
    monkeypatch.setenv("REDIS_URL", "redis://unused")
    monkeypatch.setattr("worker_scheduler.jobs._create_redis_client", lambda _url: fake)

    payload_obj = (FIX / "msg_X.json").read_text(encoding="utf-8")
    payload_str = (FIX / "msg_X_TRADE.json").read_text(encoding="utf-8")
    fake.xadd("ssi:X", {"ts_recv_utc": "1", "payload": payload_obj, "rtype": "X"})
    fake.xadd("ssi:X-TRADE", {"ts_recv_utc": "2", "payload": payload_str, "rtype": "X-TRADE"})

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        out = consume_ssi_stream_to_bronze_silver(session)
        assert out["processed"] == 2

        quotes = session.exec(select(QuoteL2)).all()
        trades = session.exec(select(TradeTape)).all()
        assert len(quotes) == 1
        assert len(trades) >= 1
