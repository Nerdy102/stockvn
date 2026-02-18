from __future__ import annotations

from core.db.models import AlertEvent, AlertRule
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from api_fastapi.deps import enforce_time_series_bounds, get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


class CreateRuleRequest(BaseModel):
    name: str
    timeframe: str = "1D"
    expression: str
    symbols: list[str] | None = None


@router.get("/rules", response_model=list[AlertRule])
def list_rules(
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[AlertRule]:
    q = select(AlertRule).order_by(AlertRule.created_at.desc()).offset(offset).limit(limit)
    return list(db.exec(q).all())


@router.post("/rules", response_model=AlertRule)
def create_rule(payload: CreateRuleRequest, db: Session = Depends(get_db)) -> AlertRule:
    symbols_csv = ",".join(payload.symbols) if payload.symbols else ""
    r = AlertRule(
        name=payload.name,
        timeframe=payload.timeframe,
        expression=payload.expression,
        symbols_csv=symbols_csv,
        is_active=True,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


@router.get("/events", response_model=list[AlertEvent])
def list_events(
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    cursor: str | None = Query(default=None, description="pagination cursor (integer offset token)"),
    db: Session = Depends(get_db),
) -> list[AlertEvent]:
    import datetime as dt

    s = dt.datetime.fromisoformat(start) if start else None
    e = dt.datetime.fromisoformat(end) if end else None
    offset = enforce_time_series_bounds(start=s, end=e, cursor=cursor, max_days=365)

    q = select(AlertEvent)
    if s:
        q = q.where(AlertEvent.triggered_at >= s)
    if e:
        q = q.where(AlertEvent.triggered_at <= e)
    q = q.order_by(AlertEvent.triggered_at.desc()).offset(offset).limit(limit)
    return list(db.exec(q).all())
