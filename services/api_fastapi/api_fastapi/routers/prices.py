from __future__ import annotations

import datetime as dt

from core.db.models import PriceOHLCV
from core.settings import Settings, get_settings
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import enforce_time_series_bounds, get_db

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
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=2000),
    cursor: str | None = Query(default=None, description="pagination cursor (integer offset token)"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[PriceOHLCV]:
    if limit > settings.API_MAX_LIMIT:
        raise HTTPException(
            status_code=400, detail=f"limit exceeds API_MAX_LIMIT={settings.API_MAX_LIMIT}"
        )

    start_dt = _parse_dt(start) if start else None
    end_dt = _parse_dt(end) if end else None
    offset = enforce_time_series_bounds(start=start_dt, end=end_dt, cursor=cursor, max_days=365)

    q = (
        select(PriceOHLCV)
        .where(PriceOHLCV.symbol == symbol)
        .where(PriceOHLCV.timeframe == timeframe)
    )
    if start_dt:
        q = q.where(PriceOHLCV.timestamp >= start_dt)
    if end_dt:
        q = q.where(PriceOHLCV.timestamp <= end_dt)
    q = q.order_by(PriceOHLCV.timestamp).offset(offset).limit(limit)
    return list(db.exec(q).all())
