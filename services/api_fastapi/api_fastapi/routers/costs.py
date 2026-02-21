from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from api_fastapi.deps import get_db
from core.costs.calibration import run_cost_calibration

router = APIRouter(prefix="/costs", tags=["costs"])


@router.post("/calibrate")
def calibrate_costs(db: Session = Depends(get_db)) -> dict[str, Any]:
    out = run_cost_calibration(db)
    if out.get("status") == "SKIP":
        raise HTTPException(status_code=400, detail={"message": out.get("message", "Không thể hiệu chỉnh."), "reason_code": "CALIBRATION_DISABLED"})
    return out
