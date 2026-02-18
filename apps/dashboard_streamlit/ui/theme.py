from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_FLAG = "_ui_theme_applied"


def apply_theme() -> None:
    if st.session_state.get(_CSS_FLAG):
        return
    css_path = Path(__file__).resolve().parent.parent / "assets" / "theme.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_CSS_FLAG] = True
