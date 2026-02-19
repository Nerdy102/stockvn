from __future__ import annotations

from fastapi import HTTPException


class LiveBrokerStub:
    def place_order(self, order: dict):
        del order
        raise HTTPException(
            status_code=501,
            detail={
                "message": "LiveBroker chưa được cấu hình an toàn. Vui lòng chạy sandbox PASS trước khi bật live.",
                "reason_code": "LIVE_BROKER_NOT_IMPLEMENTED",
            },
        )
