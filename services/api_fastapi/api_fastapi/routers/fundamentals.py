from __future__ import annotations

import datetime as dt

from core.db.models import Fundamental
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import enforce_time_series_bounds, get_db

router = APIRouter(tags=["fundamentals"])


@router.get("/fundamentals", response_model=list[Fundamental])
def list_fundamentals(
    symbol: str | None = Query(default=None),
    start: str | None = Query(default=None, description="YYYY-MM-DD as_of_date start"),
    end: str | None = Query(default=None, description="YYYY-MM-DD as_of_date end"),
    limit: int = Query(default=500, ge=1, le=2000),
    cursor: str | None = Query(default=None, description="pagination cursor (integer offset token)"),
    db: Session = Depends(get_db),
) -> list[Fundamental]:
    s = dt.date.fromisoformat(start) if start else None
    e = dt.date.fromisoformat(end) if end else None
    offset = enforce_time_series_bounds(start=s, end=e, cursor=cursor, max_days=365)

    q = select(Fundamental)
    if symbol:
        q = q.where(Fundamental.symbol == symbol)
    if s:
        q = q.where(Fundamental.as_of_date >= s)
    if e:
        q = q.where(Fundamental.as_of_date <= e)
    q = q.order_by(Fundamental.as_of_date).offset(offset).limit(limit)
    return list(db.exec(q).all())
