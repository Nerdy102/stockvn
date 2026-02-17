from __future__ import annotations

from core.db.models import Signal
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["signals"])


@router.get("/signals", response_model=list[Signal])
def list_signals(
    symbol: str | None = Query(default=None),
    timeframe: str = Query("1D"),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[Signal]:
    q = select(Signal).where(Signal.timeframe == timeframe)
    if symbol:
        q = q.where(Signal.symbol == symbol)
    q = q.order_by(Signal.timestamp.desc()).offset(offset).limit(limit)
    return list(db.exec(q).all())
