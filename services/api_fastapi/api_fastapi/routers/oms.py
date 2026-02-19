from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from core.observability.simple_metrics import (
    inc_blocked,
    inc_broker_error,
    inc_created,
    inc_executed,
    now_ms,
    observe_api_latency,
    snapshot,
)
from core.oms.models import Order, OrderEvent
from core.risk.controls_models import TradingControl
from core.oms.service import (
    approve,
    cancel,
    create_draft,
    handle_ack,
    handle_fill,
    send,
)
from core.settings import Settings, get_settings
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/oms", tags=["oms"])
log = logging.getLogger(__name__)


class OrderDraftIn(BaseModel):
    user_id: str = "guest"
    market: str = "vn"
    symbol: str
    timeframe: str = "1D"
    mode: str = "paper"
    order_type: str = "limit"
    side: str
    qty: float
    price: float | None = None
    model_id: str = ""
    config_hash: str = ""
    dataset_hash: str = ""
    code_hash: str = ""
    ts_bucket: str | None = None


class ApproveIn(BaseModel):
    order_id: str
    confirm_token: str
    checkboxes: dict[str, bool] = Field(default_factory=dict)


class ExecuteIn(BaseModel):
    order_id: str
    outside_session_vn: bool = False
    data_freshness: dict[str, Any] = Field(default_factory=dict)
    drift_alerts: dict[str, Any] = Field(default_factory=dict)
    portfolio_snapshot: dict[str, Any] = Field(default_factory=lambda: {"cash": 1_000_000_000.0, "nav_est": 1_000_000_000.0, "orders_today": 0})
    user_age: int = 30
    sandbox_passed: bool = False


class CancelIn(BaseModel):
    order_id: str




def _db_control(db: Session) -> TradingControl:
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

@router.post("/draft")
def create_order_draft(payload: OrderDraftIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    t0 = now_ms()
    order = create_draft(payload.model_dump(), db)
    inc_created()
    observe_api_latency(now_ms() - t0)
    log.info("oms_transition", extra={"event": "oms_transition", "correlation_id": order.id})
    return {"message": "Tạo lệnh nháp thành công.", "order": order.model_dump(), "reason_code": ""}


@router.post("/approve")
def approve_order(payload: ApproveIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    order = approve(payload.order_id, payload.confirm_token, payload.checkboxes, db)
    return {"message": "Duyệt lệnh thành công.", "order": order.model_dump(), "reason_code": ""}


@router.post("/execute")
def execute_order(
    payload: ExecuteIn,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    t0 = now_ms()
    order = db.get(Order, payload.order_id)
    if order is None:
        raise HTTPException(status_code=404, detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"})

    if order.mode == "live" and payload.user_age < 18:
        inc_blocked("AGE_RESTRICTED")
        raise HTTPException(status_code=403, detail={"message": "Tài khoản dưới 18 tuổi chỉ được Paper/Draft.", "reason_code": "AGE_RESTRICTED"})
    if order.mode == "live" and not settings.ENABLE_LIVE_TRADING:
        inc_blocked("LIVE_DISABLED")
        raise HTTPException(status_code=403, detail={"message": "Live trading đang bị tắt.", "reason_code": "LIVE_DISABLED"})

    control = _db_control(db)
    if settings.KILL_SWITCH or control.kill_switch_enabled:
        inc_blocked("KILL_SWITCH_ON")
        log.warning("risk_blocked", extra={"event": "risk_blocked", "correlation_id": order.id, "reason_code": "KILL_SWITCH_ON"})
        raise HTTPException(status_code=403, detail={"message": "Kill-switch đang bật, chặn mọi execute.", "reason_code": "KILL_SWITCH_ON"})
    if bool(payload.drift_alerts.get("drift_paused", False)) or (control.paused_reason_code or "").upper() == "DRIFT_PAUSED":
        inc_blocked("PAUSED_BY_SYSTEM")
        log.warning("risk_blocked", extra={"event": "risk_blocked", "correlation_id": order.id, "reason_code": "PAUSED_BY_SYSTEM"})
        raise HTTPException(status_code=403, detail={"message": "Hệ thống đang tạm dừng do drift hoặc operator.", "reason_code": "PAUSED_BY_SYSTEM"})

    risk_input = payload.model_dump(include={"data_freshness", "drift_alerts", "portfolio_snapshot", "outside_session_vn", "sandbox_passed"})
    risk_input["order_overrides"] = {"outside_session_vn": payload.outside_session_vn}
    log.info("execute_started", extra={"event": "execute_started", "correlation_id": order.id})
    try:
        sent = send(order.id, db, settings, risk_input)
    except HTTPException as exc:
        reason = (exc.detail or {}).get("reason_code", "UNKNOWN") if isinstance(exc.detail, dict) else "UNKNOWN"
        inc_blocked(str(reason))
        observe_api_latency(now_ms() - t0)
        raise
    except Exception:
        inc_broker_error()
        observe_api_latency(now_ms() - t0)
        raise
    acked = handle_ack(sent.id, {"status": "ACKED"}, db)
    filled = handle_fill(
        acked.id,
        {
            "fill_qty": acked.qty,
            "fill_price": acked.price or 0.0,
            "broker_fill_id": f"auto-fill-{acked.id[:8]}",
            "cash": float(payload.portfolio_snapshot.get("cash", 0.0)) - float(acked.qty) * float(acked.price or 0.0),
            "positions_json": {acked.symbol: acked.qty},
            "nav_est": float(payload.portfolio_snapshot.get("nav_est", 0.0)),
            "drawdown_est": 0.0,
        },
        db,
    )
    inc_executed()
    observe_api_latency(now_ms() - t0)
    log.info("execute_completed", extra={"event": "execute_completed", "correlation_id": order.id})
    return {"message": "Thực thi lệnh thành công (sandbox/paper).", "order": filled.model_dump(), "reason_code": ""}


@router.post("/cancel")
def cancel_order(payload: CancelIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = cancel(payload.order_id, db)
    return {"message": "Đã hủy lệnh.", "order": row.model_dump(), "reason_code": ""}


@router.get("/orders")
def list_orders(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.exec(select(Order).order_by(Order.created_at.desc()).limit(limit)).all()
    return {"message": "Danh sách lệnh OMS.", "items": [r.model_dump() for r in rows]}


@router.get("/orders/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.get(Order, order_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"})
    return {"message": "Chi tiết lệnh.", "item": row.model_dump()}


@router.get("/audit")
def get_audit(limit: int = Query(default=200, ge=1, le=500), db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.exec(select(OrderEvent).order_by(OrderEvent.ts.desc()).limit(limit)).all()
    return {"message": "Nhật ký audit OMS (append-only).", "items": [r.model_dump() for r in rows], "generated_at": dt.datetime.utcnow().isoformat()}


@router.get("/metrics/simple")
def simple_metrics() -> dict[str, Any]:
    return snapshot()
