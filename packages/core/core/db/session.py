from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine


@lru_cache(maxsize=2)
def get_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(database_url, echo=False, connect_args=connect_args)


def create_db_and_tables(database_url: str) -> None:
    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)


def session_scope(database_url: str) -> Iterator[Session]:
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session
