from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from api_fastapi.deps import get_db
from core.db.models import AlertEvent, AlertRule

router = APIRouter(prefix="/alerts", tags=["alerts"])


class CreateRuleRequest(BaseModel):
    name: str
    timeframe: str = "1D"
    expression: str
    symbols: Optional[List[str]] = None


@router.get("/rules", response_model=list[AlertRule])
def list_rules(db: Session = Depends(get_db)) -> List[AlertRule]:
    q = select(AlertRule).order_by(AlertRule.created_at.desc())
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
def list_events(db: Session = Depends(get_db)) -> List[AlertEvent]:
    q = select(AlertEvent).order_by(AlertEvent.triggered_at.desc())
    return list(db.exec(q).all())[:500]
