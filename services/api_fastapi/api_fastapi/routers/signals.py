from __future__ import annotations

import datetime as dt

from core.db.models import Signal
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import enforce_time_series_bounds, get_db

router = APIRouter(tags=["signals"])


@router.get("/signals", response_model=list[Signal])
def list_signals(
    symbol: str | None = Query(default=None),
    timeframe: str = Query("1D"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    cursor: str | None = Query(default=None, description="pagination cursor (integer offset token)"),
    db: Session = Depends(get_db),
) -> list[Signal]:
    s = dt.datetime.fromisoformat(start) if start else None
    e = dt.datetime.fromisoformat(end) if end else None
    offset = enforce_time_series_bounds(start=s, end=e, cursor=cursor, max_days=365)

    q = select(Signal).where(Signal.timeframe == timeframe)
    if symbol:
        q = q.where(Signal.symbol == symbol)
    if s:
        q = q.where(Signal.timestamp >= s)
    if e:
        q = q.where(Signal.timestamp <= e)
    q = q.order_by(Signal.timestamp.desc()).offset(offset).limit(limit)
    return list(db.exec(q).all())
