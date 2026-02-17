from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db
from core.db.models import Ticker

router = APIRouter(tags=["tickers"])


@router.get("/tickers", response_model=list[Ticker])
def list_tickers(
    exchange: Optional[str] = Query(default=None),
    sector: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> List[Ticker]:
    q = select(Ticker)
    if exchange:
        q = q.where(Ticker.exchange == exchange)
    if sector:
        q = q.where(Ticker.sector == sector)
    return list(db.exec(q).all())
