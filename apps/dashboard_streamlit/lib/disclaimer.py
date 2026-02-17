from __future__ import annotations

import streamlit as st

DISCLAIMER_TEXT = (
    "Quá khứ không đảm bảo tương lai; kết quả có thể overfit; "
    "phụ thuộc chi phí/độ khớp và có rủi ro."
)


def render_global_disclaimer() -> None:
    st.caption(f"⚠️ {DISCLAIMER_TEXT}")
