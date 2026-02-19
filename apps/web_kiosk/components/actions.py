from __future__ import annotations

from typing import Any

import streamlit as st

from apps.dashboard_streamlit.lib import api


def _suggestion_from_reason(reason_code: str) -> str:
    mapping = {
        "KILL_SWITCH_ON": "Gợi ý: kiểm tra và tắt kill-switch nếu đã an toàn.",
        "PAUSED_BY_SYSTEM": "Gợi ý: kiểm tra drift/reconcile trước khi resume.",
        "DATA_STALE": "Gợi ý: đồng bộ dữ liệu rồi thử lại.",
        "RISK_NOTIONAL_EXCEEDED": "Gợi ý: giảm khối lượng lệnh.",
        "RISK_CASH_BUFFER": "Gợi ý: giảm khối lượng hoặc tăng tiền mặt.",
    }
    return mapping.get(reason_code, "Gợi ý: xem audit và điều chỉnh cấu hình rủi ro.")


def render_action_bar(as_of_date: str) -> None:
    st.subheader("Thanh hành động (Action bar)")
    signal = st.session_state.get("selected_signal")
    has_signal = bool(signal)
    c1, c2 = st.columns(2)

    if c1.button("Tạo lệnh nháp (Create draft)", type="primary", use_container_width=True, disabled=not has_signal):
        payload = {
            "user_id": "kiosk-user",
            "market": "vn",
            "symbol": signal["symbol"],
            "timeframe": "1D",
            "mode": "paper",
            "order_type": "limit",
            "side": signal["side"],
            "qty": 100,
            "price": 10000,
            "model_id": signal.get("model_id", "model_1"),
            "config_hash": "kiosk-v3",
            "reason_short": signal.get("reason_short", ""),
            "risk_tags_json": {"risk_tags": signal.get("risk_tags", [])},
        }
        out = api.post("/oms/draft", payload)
        st.session_state["oms_draft"] = out.get("order")
        st.success("Tạo lệnh nháp thành công.")

    draft = st.session_state.get("oms_draft")
    ack_loss = st.checkbox("Tôi hiểu có thể thua lỗ (Risk of loss)", key="kiosk_v3_ack_loss")
    ack_edu = st.checkbox(
        "Tôi hiểu đây không phải lời khuyên đầu tư (Not investment advice)", key="kiosk_v3_ack_edu"
    )
    can_confirm = bool(draft) and ack_loss and ack_edu

    if c2.button(
        "Xác nhận thực hiện (Confirm execute)",
        use_container_width=True,
        disabled=not can_confirm,
    ):
        try:
            approve = api.post(
                "/oms/approve",
                {
                    "order_id": draft["id"],
                    "confirm_token": draft["confirm_token"],
                    "checkboxes": {"risk": ack_loss, "edu": ack_edu},
                },
            )
            execute = api.post(
                "/oms/execute",
                {
                    "order_id": draft["id"],
                    "data_freshness": {"as_of_date": as_of_date},
                    "portfolio_snapshot": {"cash": 2_000_000_000.0, "nav_est": 2_000_000_000.0, "orders_today": 0},
                    "drift_alerts": {"drift_paused": False, "kill_switch_on": False},
                },
            )
            st.success(f"Kết quả: {execute.get('order', {}).get('status', 'FILLED')}")
            st.caption(f"Audit: /oms/audit • Order ID: {draft['id']}")
            st.json({"approve": approve, "execute": execute})
        except Exception as exc:
            reason_code = "UNKNOWN"
            detail = str(exc)
            if hasattr(exc, "response") and getattr(exc, "response", None) is not None:
                try:
                    body = exc.response.json()
                    detail = str(body)
                    reason_code = str((body.get("detail") or {}).get("reason_code", "UNKNOWN"))
                except Exception:
                    pass
            st.error(f"Bị chặn/thất bại: {detail}")
            st.warning(_suggestion_from_reason(reason_code))
