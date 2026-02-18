from __future__ import annotations

import datetime as dt
from pathlib import Path

from core.db.models import BronzeFile
from sqlmodel import Session, SQLModel, create_engine, select
from worker_scheduler.jobs import bronze_retention_cleanup


def test_bronze_retention_cleanup_deletes_old_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRONZE_RETENTION_DAYS", "30")
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    old_path = tmp_path / "old.jsonl.zst"
    new_path = tmp_path / "new.jsonl.zst"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")

    now = dt.datetime.now(dt.timezone.utc)
    with Session(engine) as session:
        session.add(
            BronzeFile(
                provider="p",
                channel="c",
                date=(now - dt.timedelta(days=31)).date(),
                hour=1,
                filepath=str(old_path),
                rows=1,
                sha256="x",
            )
        )
        session.add(
            BronzeFile(
                provider="p",
                channel="c",
                date=(now - dt.timedelta(days=1)).date(),
                hour=2,
                filepath=str(new_path),
                rows=1,
                sha256="y",
            )
        )
        session.commit()

        deleted = bronze_retention_cleanup(session)
        assert deleted == 1
        assert not old_path.exists()
        assert new_path.exists()
        remain = session.exec(select(BronzeFile)).all()
        assert len(remain) == 1
        assert Path(remain[0].filepath) == new_path


def test_bronze_retention_cleanup_invalid_env_uses_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRONZE_RETENTION_DAYS", "not-an-int")
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    old_path = tmp_path / "old2.jsonl.zst"
    old_path.write_bytes(b"old")
    now = dt.datetime.now(dt.timezone.utc)

    with Session(engine) as session:
        session.add(
            BronzeFile(
                provider="p",
                channel="c",
                date=(now - dt.timedelta(days=40)).date(),
                hour=1,
                filepath=str(old_path),
                rows=1,
                sha256="x",
            )
        )
        session.commit()

        deleted = bronze_retention_cleanup(session)
        assert deleted == 1
        assert not old_path.exists()
