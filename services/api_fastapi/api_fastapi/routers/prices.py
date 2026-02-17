from __future__ import annotations

import datetime as dt
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db
from core.db.models import PriceOHLCV

router = APIRouter(tags=["prices"])


def _parse_dt(s: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return dt.datetime.fromisoformat(s + "T00:00:00")


@router.get("/prices", response_model=list[PriceOHLCV])
def get_prices(
    symbol: str = Query(...),
    timeframe: str = Query("1D"),
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> List[PriceOHLCV]:
    q = (
        select(PriceOHLCV)
        .where(PriceOHLCV.symbol == symbol)
        .where(PriceOHLCV.timeframe == timeframe)
    )
    if start:
        q = q.where(PriceOHLCV.timestamp >= _parse_dt(start))
    if end:
        q = q.where(PriceOHLCV.timestamp <= _parse_dt(end))
    q = q.order_by(PriceOHLCV.timestamp)
    return list(db.exec(q).all())
