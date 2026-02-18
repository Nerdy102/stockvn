from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "portfolio"
PAGE_TITLE = "Portfolio"


def render() -> None:
    portfolios = cached_get_json("/portfolio", params=None, ttl_s=300)
    if not portfolios and st.button("Create demo portfolio"):
        cached_post_json("/portfolio", payload={"name": "Demo Portfolio"}, ttl_s=60)
        portfolios = cached_get_json("/portfolio", params=None, ttl_s=300)

    if not portfolios:
        st.warning("Chưa có portfolio.")
        return

    pid = st.selectbox(
        "Portfolio", options=[p["id"] for p in portfolios], format_func=lambda x: f"#{x}"
    )
    if st.button("Refresh summary"):
        s = cached_get_json(f"/portfolio/{pid}/summary", params=None, ttl_s=300)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash now", value=f"{s.get('cash_now', 0):,.0f} VND")
        c2.metric("TWR", value=f"{s.get('twr', 0) * 100:.2f}%")
        c3.metric(
            "Max Drawdown", value=f"{(s.get('risk') or {}).get('max_drawdown', 0) * 100:.2f}%"
        )
        st.dataframe(pd.DataFrame(s.get("positions", [])), use_container_width=True)
