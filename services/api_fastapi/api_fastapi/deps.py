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


import datetime as dt

from fastapi import HTTPException


def parse_cursor(cursor: str | None) -> int:
    if cursor is None:
        raise HTTPException(status_code=400, detail="cursor is required when start/end are not provided")
    try:
        value = int(cursor)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="cursor must be an integer offset token") from exc
    if value < 0:
        raise HTTPException(status_code=400, detail="cursor must be >= 0")
    return value


def enforce_time_series_bounds(
    start: dt.date | dt.datetime | None,
    end: dt.date | dt.datetime | None,
    cursor: str | None,
    max_days: int = 365,
) -> int:
    if start is not None and end is not None:
        if end < start:
            raise HTTPException(status_code=400, detail="end must be >= start")
        if (end - start).days > max_days:
            raise HTTPException(status_code=400, detail=f"max range {max_days} trading days")
        return 0
    return parse_cursor(cursor)
