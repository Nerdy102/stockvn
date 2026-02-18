from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

from core.db.models import Fill, GovernanceState, Order, ReconciliationRun
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.observability.slo import INCIDENT_RULES, load_snapshots
from core.oms.order_state_machine import apply_transition
from core.oms.order_validator import validate_pretrade
from core.settings import get_settings
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["orders"])


class OrderSubmitIn(BaseModel):
    portfolio_id: int
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str = "LIMIT"
    adapter: str = "paper"


@router.post("/orders/submit")
def submit_order(payload: OrderSubmitIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    if str(payload.adapter).lower() != "paper":
        raise HTTPException(status_code=400, detail="paper adapter only")

    existing = db.exec(select(Order).where(Order.client_order_id == payload.client_order_id)).first()
    if existing is not None:
        return {"idempotent": True, "order": existing.model_dump()}

    settings = get_settings()
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)

    gov = db.exec(select(GovernanceState).order_by(GovernanceState.updated_at.desc())).first()
    if gov is not None and str(gov.status).upper() == "PAUSED":
        raise HTTPException(status_code=409, detail=f"governance paused: {gov.pause_reason}")

    idem = hashlib.sha1(
        f"{payload.portfolio_id}|{payload.client_order_id}|{payload.symbol}|{payload.side}|{payload.quantity}|{payload.price}".encode("utf-8")
    ).hexdigest()

    order = Order(
        portfolio_id=payload.portfolio_id,
        client_order_id=payload.client_order_id,
        symbol=payload.symbol,
        side=payload.side.upper(),
        quantity=payload.quantity,
        price=payload.price,
        order_type=payload.order_type,
        adapter="paper",
        state="NEW",
        idempotent_key=idem,
    )
    db.add(order)
    db.flush()

    checks = validate_pretrade(
        session=db,
        portfolio_id=payload.portfolio_id,
        symbol=payload.symbol,
        side=payload.side,
        quantity=payload.quantity,
        price=payload.price,
        as_of=dt.datetime.utcnow(),
        market_rules=mr,
        fees=fees,
        broker_name=settings.BROKER_NAME,
    )

    if not checks["ok"]:
        order.state = apply_transition(order.state, "REJECTED")
        order.reject_reason = ",".join(checks["reasons"])
        order.updated_at = dt.datetime.utcnow()
        db.commit()
        db.refresh(order)
        return {"idempotent": False, "order": order.model_dump(), "checks": checks}

    order.state = apply_transition(order.state, "VALIDATED")
    order.state = apply_transition(order.state, "SUBMITTED")
    order.updated_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(order)
    return {"idempotent": False, "order": order.model_dump(), "checks": checks}


@router.get("/orders")
def list_orders(
    portfolio_id: int | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    q = select(Order).order_by(Order.created_at.desc()).limit(limit)
    if portfolio_id is not None:
        q = q.where(Order.portfolio_id == portfolio_id)
    return [r.model_dump() for r in db.exec(q).all()]


@router.get("/fills")
def list_fills(
    order_id: int | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    q = select(Fill).order_by(Fill.filled_at.desc()).limit(limit)
    if order_id is not None:
        q = q.where(Fill.order_id == order_id)
    return [r.model_dump() for r in db.exec(q).all()]


@router.get("/governance/status")
def governance_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    gov = db.exec(select(GovernanceState).order_by(GovernanceState.updated_at.desc())).first()
    last_recon = db.exec(select(ReconciliationRun).order_by(ReconciliationRun.created_at.desc())).first()

    snapshots = load_snapshots([
        "artifacts/metrics/gateway_metrics.json",
        "artifacts/metrics/bar_builder_metrics.json",
        "artifacts/metrics/signal_engine_metrics.json",
    ])
    gauges = {
        "ingest_lag_s_p95": max((float(s.get("ingest_lag_s_p95", 0.0)) for s in snapshots), default=0.0),
        "bar_build_latency_s_p95": max((float(s.get("bar_build_latency_s_p95", 0.0)) for s in snapshots), default=0.0),
        "signal_latency_s_p95": max((float(s.get("signal_latency_s_p95", 0.0)) for s in snapshots), default=0.0),
        "redis_stream_pending": max((float(s.get("redis_stream_pending", 0.0)) for s in snapshots), default=0.0),
    }

    base = {
        "status": "RUNNING" if gov is None else gov.status,
        "pause_reason": "" if gov is None else gov.pause_reason,
        "updated_at": None if gov is None else (gov.updated_at.isoformat() if gov.updated_at else None),
        "last_reconciliation": last_recon.model_dump() if last_recon else None,
        "realtime_ops": {
            "gauges": gauges,
            "runbooks": {r.code: r.runbook_id for r in INCIDENT_RULES},
        },
    }
    return base
