from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlmodel import Session

from api_fastapi.deps import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    ok = True
    try:
        db.exec(text("SELECT 1"))
    except Exception:
        ok = False
    return {"status": "ok", "db": "ok" if ok else "error"}
