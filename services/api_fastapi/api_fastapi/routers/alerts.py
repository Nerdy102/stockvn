from __future__ import annotations

import datetime as dt

from core.calendar_vn import get_trading_calendar_vn
from core.db.models import AlertAction, AlertEvent, AlertRule, AlertV5, NotificationLog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from api_fastapi.deps import enforce_time_series_bounds, get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


class CreateRuleRequest(BaseModel):
    name: str
    timeframe: str = "1D"
    expression: str
    symbols: list[str] | None = None


class AlertActionRequest(BaseModel):
    action: str
    snooze_until: str | None = None


def _trading_days_since(d: dt.date, end: dt.date | None = None) -> int:
    cal = get_trading_calendar_vn()
    today = end or dt.date.today()
    if today <= d:
        return 0
    return max(0, len(cal.trading_days_between(d, today, inclusive="right")))


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


@router.get("/v5")
def alerts_v5_board(db: Session = Depends(get_db)) -> dict:
    rows = db.exec(select(AlertV5).order_by(AlertV5.severity.desc(), AlertV5.date.asc())).all()
    today = dt.date.today()
    board: dict[str, list[dict]] = {"NEW": [], "ACK": [], "RESOLVED": []}
    for r in rows:
        state = str(r.state).upper()
        state = state if state in board else "NEW"
        board[state].append(
            {
                "id": int(r.id or 0),
                "symbol": r.symbol,
                "date": str(r.date),
                "state": state,
                "severity": int(r.severity),
                "snooze_until": str(r.snooze_until) if r.snooze_until else None,
                "sla_escalated": bool(r.sla_escalated),
                "sla_timer_trading_days": _trading_days_since(r.date, today),
                "reason_json": r.reason_json or {},
            }
        )
    return {"as_of_date": str(today), "states": board}


@router.post("/v5/{alert_id}/action")
def alerts_v5_action(alert_id: int, payload: AlertActionRequest, db: Session = Depends(get_db)) -> dict:
    row = db.exec(select(AlertV5).where(AlertV5.id == alert_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="alert not found")

    action = str(payload.action).upper()
    old_state = str(row.state).upper()
    new_state = old_state

    if action == "ACK":
        if old_state != "NEW":
            raise HTTPException(status_code=400, detail="ACK only allowed from NEW")
        new_state = "ACK"
        row.state = new_state
    elif action == "RESOLVE":
        if old_state not in {"NEW", "ACK"}:
            raise HTTPException(status_code=400, detail="RESOLVE only allowed from NEW/ACK")
        new_state = "RESOLVED"
        row.state = new_state
    elif action == "SNOOZE":
        if old_state not in {"NEW", "ACK"}:
            raise HTTPException(status_code=400, detail="SNOOZE only allowed from NEW/ACK")
        if not payload.snooze_until:
            raise HTTPException(status_code=400, detail="snooze_until required")
        row.snooze_until = dt.datetime.strptime(payload.snooze_until, "%Y-%m-%d").date()
    else:
        raise HTTPException(status_code=400, detail="unsupported action")

    row.updated_at = dt.datetime.utcnow()
    db.add(row)
    db.add(
        AlertAction(
            alert_id=int(row.id or 0),
            action=action,
            payload_json={"from": old_state, "to": new_state, "snooze_until": payload.snooze_until},
        )
    )
    db.commit()

    return {"ok": True, "alert_id": alert_id, "state": row.state, "snooze_until": str(row.snooze_until) if row.snooze_until else None}


@router.get("/v5/digest-log")
def alerts_v5_digest_log(limit: int = Query(default=50, ge=1, le=500), db: Session = Depends(get_db)) -> list[dict]:
    rows = db.exec(select(NotificationLog).where(NotificationLog.kind == "alert_digest_v5").order_by(NotificationLog.created_at.desc()).limit(limit)).all()
    return [
        {"id": int(r.id or 0), "channel": r.channel, "created_at": r.created_at.isoformat(), "payload_json": r.payload_json}
        for r in rows
    ]
