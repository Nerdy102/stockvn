from __future__ import annotations

import datetime as dt

from core.db.models import Fundamental
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["fundamentals"])


@router.get("/fundamentals", response_model=list[Fundamental])
def list_fundamentals(
    symbol: str | None = Query(default=None),
    start: str | None = Query(default=None, description="YYYY-MM-DD as_of_date start"),
    end: str | None = Query(default=None, description="YYYY-MM-DD as_of_date end"),
    limit: int = Query(default=500, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[Fundamental]:
    if start and end:
        s = dt.date.fromisoformat(start)
        e = dt.date.fromisoformat(end)
        if e < s:
            raise HTTPException(status_code=400, detail="end must be >= start")
        if offset == 0 and (e - s).days > 365:
            raise HTTPException(status_code=400, detail="max range 365 days without pagination")

    q = select(Fundamental)
    if symbol:
        q = q.where(Fundamental.symbol == symbol)
    if start:
        q = q.where(Fundamental.as_of_date >= dt.date.fromisoformat(start))
    if end:
        q = q.where(Fundamental.as_of_date <= dt.date.fromisoformat(end))
    q = q.order_by(Fundamental.as_of_date).offset(offset).limit(limit)
    return list(db.exec(q).all())
