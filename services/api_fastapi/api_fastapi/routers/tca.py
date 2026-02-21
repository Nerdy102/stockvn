from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import OperationalError
from sqlmodel import SQLModel, Session, select

from api_fastapi.deps import get_db
from core.tca.models import OrderTCA
from core.oms.models import Order

router = APIRouter(prefix="/tca", tags=["tca"])


@router.get("/summary")
def tca_summary(
    limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)
) -> dict[str, Any]:
    try:
        SQLModel.metadata.create_all(db.get_bind(), tables=[OrderTCA.__table__])
        rows = db.exec(select(OrderTCA).order_by(OrderTCA.created_at.desc()).limit(limit)).all()
    except OperationalError:
        rows = []
    if not rows:
        return {
            "overall": {
                "median_is_bps": 0.0,
                "p95_is_bps": 0.0,
                "good_rate": 0.0,
                "last_updated": None,
            },
            "by_model": {},
            "by_market": {},
        }
    bps = sorted(float(r.is_bps_total) for r in rows)
    median = bps[len(bps) // 2]
    p95 = bps[min(len(bps) - 1, int(0.95 * (len(bps) - 1)))]
    good = sum(1 for r in rows if str(r.quality_bucket) == "Tốt") / max(len(rows), 1)
    by_market: dict[str, list[float]] = {}
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_market.setdefault(str(r.market), []).append(float(r.is_bps_total))
        ord_row = db.get(Order, r.order_id)
        model_id = str((ord_row.model_id if ord_row is not None else "") or "không rõ")
        by_model.setdefault(model_id, []).append(float(r.is_bps_total))
    by_market_out = {k: float(sorted(v)[len(v) // 2]) for k, v in by_market.items()}
    by_model_out = {k: float(sorted(v)[len(v) // 2]) for k, v in by_model.items()}
    return {
        "overall": {
            "median_is_bps": float(median),
            "p95_is_bps": float(p95),
            "good_rate": float(good),
            "last_updated": max(r.created_at for r in rows).isoformat(),
        },
        "by_model": by_model_out,
        "by_market": by_market_out,
    }


@router.get("/orders/{order_id}")
def tca_order_detail(order_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        SQLModel.metadata.create_all(db.get_bind(), tables=[OrderTCA.__table__])
        row = db.exec(select(OrderTCA).where(OrderTCA.order_id == order_id)).first()
    except OperationalError:
        row = None
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"message": "Không tìm thấy TCA cho lệnh.", "reason_code": "TCA_NOT_FOUND"},
        )
    return {"item": row.model_dump()}
