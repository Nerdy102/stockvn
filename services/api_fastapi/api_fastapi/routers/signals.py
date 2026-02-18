from __future__ import annotations

import datetime as dt

from core.db.models import Signal
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["signals"])


@router.get("/signals", response_model=list[Signal])
def list_signals(
    symbol: str | None = Query(default=None),
    timeframe: str = Query("1D"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[Signal]:
    if start and end:
        s = dt.datetime.fromisoformat(start)
        e = dt.datetime.fromisoformat(end)
        if e < s:
            raise HTTPException(status_code=400, detail="end must be >= start")
        if offset == 0 and (e - s).days > 365:
            raise HTTPException(status_code=400, detail="max range 365 days without pagination")

    q = select(Signal).where(Signal.timeframe == timeframe)
    if symbol:
        q = q.where(Signal.symbol == symbol)
    if start:
        q = q.where(Signal.timestamp >= dt.datetime.fromisoformat(start))
    if end:
        q = q.where(Signal.timestamp <= dt.datetime.fromisoformat(end))
    q = q.order_by(Signal.timestamp.desc()).offset(offset).limit(limit)
    return list(db.exec(q).all())
