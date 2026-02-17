from __future__ import annotations

from collections.abc import Generator

from core.db.session import get_engine
from core.settings import Settings, get_settings
from fastapi import Depends
from sqlmodel import Session


def get_db(settings: Settings = Depends(get_settings)) -> Generator[Session, None, None]:
    engine = get_engine(settings.DATABASE_URL)
    with Session(engine) as session:
        yield session
