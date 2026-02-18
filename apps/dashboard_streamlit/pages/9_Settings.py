from __future__ import annotations

import streamlit as st

PAGE_ID = "settings"
PAGE_TITLE = "Settings"


def render() -> None:
    st.subheader("UI Settings")
    st.toggle("Show debug info", value=False, key="ui_show_debug")
    st.toggle("Enable realtime polling", value=False, key="realtime_enabled")
    st.caption(
        "Các thao tác đặt lệnh paper/live sẽ yêu cầu xác nhận ngôn ngữ đơn giản ở PR tiếp theo."
    )
