from __future__ import annotations

from core.db.models import Fundamental
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["fundamentals"])


@router.get("/fundamentals", response_model=list[Fundamental])
def list_fundamentals(
    symbol: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[Fundamental]:
    q = select(Fundamental)
    if symbol:
        q = q.where(Fundamental.symbol == symbol)
    q = q.order_by(Fundamental.as_of_date)
    return list(db.exec(q).all())
