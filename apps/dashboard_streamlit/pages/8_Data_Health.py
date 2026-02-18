from __future__ import annotations

import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json

PAGE_ID = "data_health"
PAGE_TITLE = "Data Health"


def render() -> None:
    st.subheader("Data health summary")
    try:
        st.json(cached_get_json("/data/health/summary", params=None, ttl_s=60))
    except Exception:
        st.json(cached_get_json("/health", params=None, ttl_s=60))
