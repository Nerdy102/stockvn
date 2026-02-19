from __future__ import annotations

import datetime as dt
from typing import Any

FIXED_REASON_CODES = {
    "RISK_NOTIONAL_EXCEEDED",
    "RISK_DAILY_ORDER_LIMIT",
    "RISK_CASH_BUFFER",
    "SESSION_OFF_HOURS",
    "DATA_STALE",
    "DRIFT_PAUSED",
    "KILL_SWITCH_ON",
}


def evaluate_pretrade(
    order: dict[str, Any],
    portfolio_snapshot: dict[str, Any],
    data_freshness: dict[str, Any],
    drift_alerts: dict[str, Any],
) -> tuple[bool, str, str]:
    if bool(drift_alerts.get("kill_switch_on", False)):
        return False, "KILL_SWITCH_ON", "Kill switch đang bật, tạm dừng thực thi lệnh."

    if bool(drift_alerts.get("drift_paused", False)):
        return False, "DRIFT_PAUSED", "Hệ thống đang tạm dừng do cảnh báo drift."

    nav = float(portfolio_snapshot.get("nav_est", 0.0))
    cash = float(portfolio_snapshot.get("cash", 0.0))
    order_notional = float(order.get("qty", 0.0)) * float(order.get("price") or 0.0)

    max_notional_pct = float(order.get("max_notional_per_order_pct", 0.2))
    if nav > 0 and order_notional > nav * max_notional_pct:
        return False, "RISK_NOTIONAL_EXCEEDED", "Giá trị lệnh vượt ngưỡng notional cho phép."

    max_orders_per_day = int(order.get("max_orders_per_day", 20))
    orders_today = int(portfolio_snapshot.get("orders_today", 0))
    if orders_today >= max_orders_per_day:
        return False, "RISK_DAILY_ORDER_LIMIT", "Đã vượt số lệnh tối đa trong ngày."

    min_cash_buffer_pct = float(order.get("min_cash_buffer_pct", 0.05))
    if nav > 0 and (cash - order_notional) < nav * min_cash_buffer_pct:
        return False, "RISK_CASH_BUFFER", "Không đảm bảo tỷ lệ tiền mặt dự phòng tối thiểu."

    market = str(order.get("market", "vn"))
    outside_session = bool(order.get("outside_session_vn", False)) if market == "vn" else False
    if outside_session:
        return False, "SESSION_OFF_HOURS", "Ngoài giờ giao dịch, chỉ cho phép tạo nháp."

    now = dt.datetime.utcnow()
    if market == "vn":
        as_of_date = data_freshness.get("as_of_date")
        if isinstance(as_of_date, str):
            as_of_date = dt.date.fromisoformat(as_of_date)
        if as_of_date is None or (now.date() - as_of_date).days > 2:
            return False, "DATA_STALE", "Dữ liệu VN đã stale quá 2 ngày, chặn thực thi."
    else:
        as_of_ts = data_freshness.get("as_of_ts")
        if isinstance(as_of_ts, str):
            as_of_ts = dt.datetime.fromisoformat(as_of_ts)
        if as_of_ts is None or (now - as_of_ts).total_seconds() > 6 * 3600:
            return False, "DATA_STALE", "Dữ liệu crypto stale quá 6 giờ, chặn thực thi."

    return True, "", "Đủ điều kiện risk gating để thực thi."
