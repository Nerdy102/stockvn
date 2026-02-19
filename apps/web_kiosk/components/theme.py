from __future__ import annotations

import streamlit as st

FONT_STACK_VI = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
            font-family: {FONT_STACK_VI};
        }}
        .kiosk-card {{
            border: 1px solid rgba(49, 51, 63, 0.20);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
