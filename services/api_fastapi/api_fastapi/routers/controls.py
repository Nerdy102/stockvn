from __future__ import annotations

import datetime as dt
from typing import Any

from core.risk.controls_models import TradingControl
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import SQLModel, Session

from api_fastapi.deps import get_db

router = APIRouter(prefix="/controls", tags=["controls"])


class PauseIn(BaseModel):
    reason_code: str


def _get_or_create(db: Session) -> TradingControl:
    try:
        SQLModel.metadata.create_all(db.get_bind(), tables=[TradingControl.__table__])
    except Exception:
        pass
    row = db.get(TradingControl, 1)
    if row is None:
        row = TradingControl(id=1, kill_switch_enabled=False)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


@router.post("/kill_switch/on")
def kill_switch_on(db: Session = Depends(get_db)) -> dict[str, Any]:
    row = _get_or_create(db)
    row.kill_switch_enabled = True
    row.paused_reason_code = "KILL_SWITCH_ON"
    row.updated_at = dt.datetime.utcnow()
    db.add(row)
    db.commit()
    return {"message": "Đã bật Dừng khẩn cấp (Kill-switch).", "reason_code": "KILL_SWITCH_ON"}


@router.post("/kill_switch/off")
def kill_switch_off(db: Session = Depends(get_db)) -> dict[str, Any]:
    row = _get_or_create(db)
    row.kill_switch_enabled = False
    row.paused_reason_code = None
    row.updated_at = dt.datetime.utcnow()
    db.add(row)
    db.commit()
    return {"message": "Đã tắt Dừng khẩn cấp (Kill-switch).", "reason_code": ""}


@router.post("/pause")
def pause_system(payload: PauseIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = _get_or_create(db)
    row.kill_switch_enabled = True
    row.paused_reason_code = payload.reason_code
    row.updated_at = dt.datetime.utcnow()
    db.add(row)
    db.commit()
    return {"message": "Hệ thống đã tạm dừng.", "reason_code": payload.reason_code}


@router.post("/resume")
def resume_system(db: Session = Depends(get_db)) -> dict[str, Any]:
    row = _get_or_create(db)
    row.kill_switch_enabled = False
    row.paused_reason_code = None
    row.updated_at = dt.datetime.utcnow()
    db.add(row)
    db.commit()
    return {"message": "Hệ thống đã tiếp tục hoạt động.", "reason_code": ""}
