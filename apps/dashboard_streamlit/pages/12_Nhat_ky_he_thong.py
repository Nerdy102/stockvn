from __future__ import annotations

import httpx
import streamlit as st

from apps.dashboard_streamlit.lib import api

FONT_STACK_VI = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'


def render() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
            font-family: {FONT_STACK_VI};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🧾 Nhật ký hệ thống (Audit log)")
    st.info("Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư.")
    limit = st.slider("Giới hạn bản ghi", min_value=20, max_value=500, value=200, step=20)

    try:
        logs = api.get("/simple/audit_logs", {"limit": limit})
        health = api.get("/healthz/detail", {})
        reconcile = api.get("/reconcile/latest", {})
    except (httpx.HTTPError, ValueError):
        st.error("Không thể kết nối API để đọc nhật ký hệ thống.")
        return

    st.subheader("Sức khoẻ hệ thống (System health)")
    st.write(
        f"DB: {'OK' if health.get('db_ok') else 'FAIL'} • Độ trễ DB: {health.get('db_latency_ms','N/A')}ms • Broker: {'OK' if health.get('broker_ok') else 'FAIL'} • Kill-switch: {'BẬT' if health.get('kill_switch_state') else 'TẮT'}"
    )
    st.write(
        f"Độ mới dữ liệu (Data freshness): {'OK' if health.get('data_freshness_ok') else 'STALE'} • as_of: {health.get('as_of_date','N/A')} • Drift pause: {'BẬT' if health.get('drift_pause_state') else 'TẮT'}"
    )
    st.write(
        f"Đối soát gần nhất (Last reconcile): {reconcile.get('last_reconcile_ts','N/A')} • Số mismatch: {reconcile.get('mismatch_count',0)} • Trạng thái: {reconcile.get('status','N/A')}"
    )

    st.subheader("Sự kiện kiểm toán")
    items = logs.get("items", [])
    if not items:
        st.warning(logs.get("message", "Chưa có dữ liệu"))
        return
    st.dataframe(items, use_container_width=True)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
