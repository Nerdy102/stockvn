from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache

from sqlmodel import Session, SQLModel, create_engine

from core.logging import get_logger

log = get_logger(__name__)


def get_database_url(default: str = "sqlite:///./vn_invest.db") -> str:
    return os.getenv("DATABASE_URL", default)


@lru_cache(maxsize=4)
def get_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(database_url, echo=False, connect_args=connect_args, pool_pre_ping=True)


def create_db_and_tables(database_url: str) -> None:
    """
    Development helper for SQLite.

    Production PostgreSQL schema must be managed by Alembic migrations.
    """
    if not database_url.startswith("sqlite"):
        log.info(
            "skip_create_all_non_sqlite",
            extra={"event": "db_schema", "database_url": database_url.split(":", 1)[0]},
        )
        return

    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)


def session_scope(database_url: str) -> Iterator[Session]:
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session
