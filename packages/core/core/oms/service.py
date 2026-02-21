from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
from typing import Any

from core.brokers.live_stub import LiveBrokerStub
from core.brokers.paper import PaperBroker
from core.brokers.sandbox import SandboxBroker
from core.oms.models import Fill, Order, OrderEvent, PortfolioSnapshot
from core.tca.service import compute_tca_for_order
from core.risk.gate import evaluate_pretrade
from core.settings import Settings
from fastapi import HTTPException
from sqlmodel import Session, select

_ALLOWED_TRANSITIONS = {
    "DRAFT": {"APPROVED", "CANCELLED", "ERROR"},
    "APPROVED": {"SENT", "CANCELLED", "REJECTED", "ERROR"},
    "SENT": {"ACKED", "REJECTED", "CANCELLED", "ERROR"},
    "ACKED": {"PARTIAL_FILLED", "FILLED", "REJECTED", "CANCELLED", "ERROR"},
    "PARTIAL_FILLED": {"FILLED", "CANCELLED", "ERROR"},
    "FILLED": set(),
    "REJECTED": set(),
    "CANCELLED": set(),
    "ERROR": set(),
}


def _price_bucket(price: float | None) -> str:
    return f"{float(price or 0.0):.2f}"


def _ts_bucket(now: dt.datetime | None = None) -> str:
    ref = now or dt.datetime.utcnow()
    return ref.strftime("%Y%m%d%H%M")


def build_idempotency_key(payload: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(payload.get("user_id", "")),
            str(payload.get("market", "")),
            str(payload.get("symbol", "")),
            str(payload.get("side", "")),
            str(payload.get("qty", "")),
            _price_bucket(payload.get("price")),
            str(payload.get("ts_bucket") or _ts_bucket()),
            str(payload.get("model_id", "")),
            str(payload.get("config_hash", "")),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _confirm_token(order_id: str, created_at: dt.datetime) -> str:
    secret = os.getenv("OMS_CONFIRM_SECRET", "dev-confirm-secret")
    msg = f"{order_id}|{created_at.isoformat()}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _transition(
    session: Session, order: Order, to_status: str, event_type: str, payload: dict[str, Any]
) -> None:
    if to_status not in _ALLOWED_TRANSITIONS.get(order.status, set()):
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Chuyển trạng thái không hợp lệ.",
                "reason_code": "INVALID_STATE_TRANSITION",
            },
        )
    event = OrderEvent(
        order_id=order.id,
        from_status=order.status,
        to_status=to_status,
        event_type=event_type,
        payload_json=payload,
        correlation_id=str(payload.get("correlation_id", order.id)),
    )
    order.status = to_status
    order.updated_at = dt.datetime.utcnow()
    session.add(event)


def _normalize_execution_pref(v: Any) -> str:
    t = str(v or "close").strip().lower()
    if "next" in t:
        return "next_bar"
    return "close"


def create_draft(order_draft_request: dict[str, Any], session: Session) -> Order:
    idem = build_idempotency_key(order_draft_request)
    existing = session.exec(select(Order).where(Order.idempotency_key == idem)).first()
    if existing:
        return existing

    order = Order(
        user_id=str(order_draft_request.get("user_id", "anonymous")),
        market=str(order_draft_request.get("market", "vn")),
        symbol=str(order_draft_request["symbol"]),
        timeframe=str(order_draft_request.get("timeframe", "1D")),
        mode=str(order_draft_request.get("mode", "paper")),
        order_type=str(order_draft_request.get("order_type", "market")),
        execution_pref=_normalize_execution_pref(
            order_draft_request.get("execution_pref") or order_draft_request.get("execution")
        ),
        side=str(order_draft_request["side"]).upper(),
        qty=float(order_draft_request["qty"]),
        price=order_draft_request.get("price"),
        status="DRAFT",
        idempotency_key=idem,
        client_order_id=order_draft_request.get("client_order_id"),
        notional_est=float(order_draft_request.get("notional_est") or 0.0),
        fee_est=float(order_draft_request.get("fee_est") or 0.0),
        tax_est=float(order_draft_request.get("tax_est") or 0.0),
        slippage_est=float(order_draft_request.get("slippage_est") or 0.0),
        reason_short=str(order_draft_request.get("reason_short", "")),
        risk_tags_json=dict(order_draft_request.get("risk_tags_json") or {}),
        model_id=str(order_draft_request.get("model_id", "")),
        config_hash=str(order_draft_request.get("config_hash", "")),
        dataset_hash=str(order_draft_request.get("dataset_hash", "")),
        code_hash=str(order_draft_request.get("code_hash", "")),
    )
    order.confirm_token = _confirm_token(order.id, order.created_at)
    session.add(order)
    session.add(
        OrderEvent(
            order_id=order.id,
            from_status="DRAFT",
            to_status="DRAFT",
            event_type="CREATE",
            payload_json={"sanitized": True},
            correlation_id=order.id,
        )
    )
    session.commit()
    session.refresh(order)
    return order


def approve(
    order_id: str, confirm_token: str, checkboxes: dict[str, bool], session: Session
) -> Order:
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"},
        )
    if order.confirm_used_at is not None:
        raise HTTPException(
            status_code=409,
            detail={"message": "Token xác nhận đã dùng.", "reason_code": "CONFIRM_ALREADY_USED"},
        )
    if confirm_token != order.confirm_token:
        raise HTTPException(
            status_code=403,
            detail={"message": "Token xác nhận không hợp lệ.", "reason_code": "CONFIRM_INVALID"},
        )
    if not all(bool(v) for v in checkboxes.values()):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cần tích đầy đủ checkbox xác nhận.",
                "reason_code": "CONFIRM_CHECKBOX_REQUIRED",
            },
        )

    _transition(session, order, "APPROVED", "APPROVE", {"checkboxes": checkboxes})
    order.confirm_used_at = dt.datetime.utcnow()
    session.commit()
    session.refresh(order)
    return order


def _pick_broker(settings: Settings, mode: str, risk_input: dict[str, Any]):
    if mode == "live":
        if (not settings.ENABLE_LIVE_TRADING) or settings.TRADING_ENV != "live":
            raise HTTPException(
                status_code=403,
                detail={"message": "Live trading đang tắt.", "reason_code": "LIVE_DISABLED"},
            )
        if settings.REQUIRE_SANDBOX_PASS_BEFORE_LIVE and not bool(
            risk_input.get("sandbox_passed", False)
        ):
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Sandbox chưa PASS, chưa được bật live.",
                    "reason_code": "SANDBOX_REQUIRED",
                },
            )
        return LiveBrokerStub()
    if mode == "paper":
        return PaperBroker()
    return SandboxBroker()


def send(order_id: str, session: Session, settings: Settings, risk_input: dict[str, Any]) -> Order:
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"},
        )

    order_payload = order.model_dump()
    order_payload.update(risk_input.get("order_overrides") or {})
    allowed, reason_code, message_vi = evaluate_pretrade(
        order=order_payload,
        portfolio_snapshot=risk_input.get("portfolio_snapshot") or {},
        data_freshness=risk_input.get("data_freshness") or {},
        drift_alerts=risk_input.get("drift_alerts") or {},
    )
    if not allowed:
        raise HTTPException(
            status_code=403, detail={"message": message_vi, "reason_code": reason_code}
        )

    broker = _pick_broker(settings, order.mode, risk_input)
    _transition(session, order, "SENT", "SEND", {"mode": order.mode})
    ack = broker.place_order(order.model_dump())
    order.broker_order_id = ack.broker_order_id
    session.commit()
    session.refresh(order)
    return order


def handle_ack(order_id: str, broker_resp: dict[str, Any], session: Session) -> Order:
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"},
        )
    _transition(
        session, order, "ACKED", "ACK", {"broker_resp": {"status": broker_resp.get("status")}}
    )
    session.commit()
    session.refresh(order)
    return order


def handle_fill(order_id: str, fill: dict[str, Any], session: Session) -> Order:
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"},
        )

    fill_qty = float(fill.get("fill_qty", 0.0))
    session.add(
        Fill(
            order_id=order_id,
            fill_qty=fill_qty,
            fill_price=float(fill.get("fill_price", order.price or 0.0)),
            fee=float(fill.get("fee", 0.0)),
            tax=float(fill.get("tax", 0.0)),
            slippage_cost=float(fill.get("slippage_cost", 0.0)),
            funding_cost=float(fill.get("funding_cost", 0.0)),
            pnl_gross=fill.get("pnl_gross"),
            pnl_net=fill.get("pnl_net"),
            broker_fill_id=fill.get("broker_fill_id"),
            correlation_id=str(fill.get("correlation_id", order_id)),
        )
    )
    to_status = "FILLED" if fill_qty >= float(order.qty) else "PARTIAL_FILLED"
    _transition(session, order, to_status, "FILL", {"fill_qty": fill_qty})
    session.add(
        PortfolioSnapshot(
            ts=dt.datetime.utcnow(),
            cash=float(fill.get("cash", 0.0)),
            positions_json=fill.get("positions_json") or {},
            nav_est=float(fill.get("nav_est", 0.0)),
            drawdown_est=float(fill.get("drawdown_est", 0.0)),
        )
    )
    session.commit()
    session.refresh(order)
    if order.status == "FILLED":
        compute_tca_for_order(session, order_id)
    return order


def handle_reject(order_id: str, reason: str, session: Session) -> Order:
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"},
        )
    _transition(session, order, "REJECTED", "REJECT", {"reason": reason})
    order.reason_short = reason
    session.commit()
    session.refresh(order)
    return order


def cancel(order_id: str, session: Session) -> Order:
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy lệnh.", "reason_code": "ORDER_NOT_FOUND"},
        )
    _transition(session, order, "CANCELLED", "CANCEL", {"cancel": True})
    session.commit()
    session.refresh(order)
    prev_events = session.exec(
        select(OrderEvent).where(OrderEvent.order_id == order_id).order_by(OrderEvent.ts.asc())
    ).all()
    if any(e.to_status == "PARTIAL_FILLED" for e in prev_events):
        compute_tca_for_order(session, order_id)
    return order
