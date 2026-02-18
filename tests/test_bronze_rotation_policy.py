from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

from core.db.models import BronzeFile
from data.bronze.writer import BronzeWriter
from sqlmodel import Session, SQLModel, create_engine, select


def test_bronze_rotation_policy_by_record_count(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        writer = BronzeWriter(
            provider="ssi_fastconnect",
            channel="ssi_stream_X",
            session=session,
            base_dir=str(tmp_path),
            rotate_every_records=2,
            rotate_every_seconds=3600,
        )
        assert writer.write({"v": 1}) is False
        assert writer.write({"v": 2}) is True

        rows = session.exec(select(BronzeFile)).all()
        assert len(rows) == 1
        assert rows[0].rows == 2


def test_bronze_rotation_policy_by_time(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    start = dt.datetime(2026, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    later = start + dt.timedelta(seconds=601)

    with Session(engine) as session:
        writer = BronzeWriter(
            provider="ssi_fastconnect",
            channel="ssi_stream_X",
            session=session,
            base_dir=str(tmp_path),
            rotate_every_records=100,
            rotate_every_seconds=600,
        )
        assert writer.write({"v": 1}, now_utc=start) is False
        assert writer.write({"v": 2}, now_utc=later) is True

        rows = session.exec(select(BronzeFile)).all()
        assert len(rows) == 1
        assert rows[0].rows == 2


def test_bronze_writer_appends_in_same_hour(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    start = dt.datetime(2026, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    with Session(engine) as session:
        writer = BronzeWriter(
            provider="ssi_fastconnect",
            channel="ssi_stream_X",
            session=session,
            base_dir=str(tmp_path),
            rotate_every_records=10,
            rotate_every_seconds=600,
        )
        writer.write({"v": 1}, now_utc=start)
        writer.flush(now_utc=start)
        writer.write({"v": 2}, now_utc=start + dt.timedelta(minutes=1))
        writer.flush(now_utc=start + dt.timedelta(minutes=1))

        rows = session.exec(select(BronzeFile)).all()
        assert len(rows) == 1
        assert rows[0].rows == 2

        import zstandard as zstd

        raw = Path(rows[0].filepath).read_bytes()
        decoded_parts: list[bytes] = []
        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                decoded_parts.append(chunk)
        decoded = b"".join(decoded_parts).decode("utf-8")
        assert '"v":1' in decoded
        assert '"v":2' in decoded


def test_bronze_writer_rotates_on_hour_boundary(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    t1 = dt.datetime(2026, 1, 1, 1, 59, 59, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 1, 1, 2, 0, 1, tzinfo=dt.timezone.utc)

    with Session(engine) as session:
        writer = BronzeWriter(
            provider="ssi_fastconnect",
            channel="ssi_stream_X",
            session=session,
            base_dir=str(tmp_path),
            rotate_every_records=100,
            rotate_every_seconds=3600,
        )
        writer.write({"v": 1}, now_utc=t1)
        writer.write({"v": 2}, now_utc=t2)
        writer.flush(now_utc=t2)

        rows = session.exec(select(BronzeFile).order_by(BronzeFile.hour)).all()
        assert len(rows) == 2
        assert rows[0].hour == 1
        assert rows[1].hour == 2
        assert rows[0].rows == 1
        assert rows[1].rows == 1


def test_bronze_writer_accepts_naive_datetime(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    naive = dt.datetime(2026, 1, 1, 3, 0, 0)
    with Session(engine) as session:
        writer = BronzeWriter(
            provider="ssi_fastconnect",
            channel="ssi_stream_X",
            session=session,
            base_dir=str(tmp_path),
            rotate_every_records=10,
            rotate_every_seconds=600,
        )
        writer.write({"v": 1}, now_utc=naive)
        writer.flush(now_utc=naive)

        row = session.exec(select(BronzeFile)).one()
        assert row.hour == 3
        assert row.rows == 1
