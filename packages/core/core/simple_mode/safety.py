from __future__ import annotations

from fastapi import HTTPException


def _raise_safety(reason_code: str, message: str) -> None:
    raise HTTPException(status_code=422, detail={"reason_code": reason_code, "message": message})


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
        _raise_safety(
            "DISCLAIMERS_NOT_ACKNOWLEDGED",
            "Cần xác nhận: đây là công cụ giáo dục và có thể thua lỗ.",
        )

    if age is not None and age < 18 and mode == "live":
        _raise_safety(
            "LIVE_BLOCKED_UNDER_18",
            "Dưới 18 tuổi chỉ dùng Draft/Paper; không hỗ trợ lách luật.",
        )

    if mode == "live":
        if not live_trading_enabled():
            _raise_safety("LIVE_DISABLED_BY_DEFAULT", "Live trading đang tắt mặc định.")
        if not acknowledged_live_eligibility:
            _raise_safety(
                "LIVE_LEGAL_ELIGIBILITY_REQUIRED",
                "Cần xác nhận đủ điều kiện pháp lý để dùng live trading.",
            )
