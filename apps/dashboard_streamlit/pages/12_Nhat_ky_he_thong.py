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
        health = api.get("/simple/system_health", {})
    except (httpx.HTTPError, ValueError):
        st.error("Không thể kết nối API để đọc nhật ký hệ thống.")
        return

    st.subheader("Sức khoẻ hệ thống (System health)")
    freshness = health.get('data_freshness', {})
    st.write(
        f"Kết nối broker: {health.get('broker_connectivity','N/A')} • Redis: {health.get('redis_connectivity','N/A')} • Kill-switch cấu hình: {'BẬT' if health.get('config_kill_switch') else 'TẮT'} • Kill-switch runtime: {'BẬT' if health.get('runtime_kill_switch') else 'TẮT'} • Kill-switch DB: {'BẬT' if health.get('db_kill_switch') else 'TẮT'}"
    )
    st.write(
        f"Độ mới dữ liệu (Data freshness): {freshness.get('status','N/A')} • Cập nhật gần nhất: {freshness.get('last_update','N/A')}"
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
