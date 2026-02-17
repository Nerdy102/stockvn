from __future__ import annotations

from core.db.models import Ticker
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["tickers"])


@router.get("/tickers", response_model=list[Ticker])
def list_tickers(
    exchange: str | None = Query(default=None),
    sector: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[Ticker]:
    q = select(Ticker)
    if exchange:
        q = q.where(Ticker.exchange == exchange.upper())
    if sector:
        q = q.where(Ticker.sector == sector)
    q = q.order_by(Ticker.symbol).offset(offset).limit(limit)
    return list(db.exec(q).all())
