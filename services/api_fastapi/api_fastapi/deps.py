from __future__ import annotations

from typing import Generator

from fastapi import Depends
from sqlmodel import Session

from core.db.session import get_engine
from core.settings import Settings, get_settings


def get_db(settings: Settings = Depends(get_settings)) -> Generator[Session, None, None]:
    engine = get_engine(settings.DATABASE_URL)
    with Session(engine) as session:
        yield session
