from __future__ import annotations

import datetime as dt

from core.db.models import UniverseAudit
from core.universe.manager import UniverseManager
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/universe", tags=["universe"])


@router.get("")
def get_universe(
    date: str = Query(..., description="dd-mm-YYYY"),
    name: str = Query(default="ALL", pattern="^(ALL|VN30|VNINDEX)$"),
    db: Session = Depends(get_db),
) -> dict:
    try:
        as_of = dt.datetime.strptime(date, "%d-%m-%Y").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be dd-mm-YYYY") from exc

    normalized_name = name.upper()
    manager = UniverseManager(db)
    symbols, breakdown = manager.universe(date=as_of, name=normalized_name)
    audit = db.exec(
        select(UniverseAudit)
        .where(UniverseAudit.date == as_of)
        .where(UniverseAudit.universe_name == normalized_name)
    ).first()

    return {
        "date": as_of.isoformat(),
        "universe": normalized_name,
        "symbols": symbols,
        "audit": {
            "included_count": int(audit.included_count if audit else len(symbols)),
            "excluded_json_breakdown": dict(audit.excluded_json_breakdown if audit else breakdown),
        },
    }
