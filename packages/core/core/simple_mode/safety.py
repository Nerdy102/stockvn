from __future__ import annotations

from fastapi import HTTPException


def live_trading_enabled() -> bool:
    import os

    return str(os.getenv("ENABLE_LIVE_TRADING", "false")).lower() == "true"


def ensure_disclaimers(
    *,
    acknowledged_educational: bool,
    acknowledged_loss: bool,
    mode: str,
    acknowledged_live_eligibility: bool,
    age: int | None,
) -> None:
    if not acknowledged_educational or not acknowledged_loss:
        raise HTTPException(status_code=422, detail="Cần xác nhận disclaimer bắt buộc")

    if age is not None and age < 18 and mode == "live":
        raise HTTPException(
            status_code=422,
            detail="Dưới 18 tuổi có thể cần người giám hộ/đủ tuổi để mở tài khoản, không hỗ trợ lách luật",
        )

    if mode == "live":
        if not live_trading_enabled():
            raise HTTPException(status_code=422, detail="Live trading đang tắt mặc định")
        if not acknowledged_live_eligibility:
            raise HTTPException(
                status_code=422, detail="Cần xác nhận đủ điều kiện pháp lý cho live"
            )
