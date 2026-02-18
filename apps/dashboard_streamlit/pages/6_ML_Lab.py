from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json
from apps.dashboard_streamlit.ui.perf import DAILY_MAX_DAYS_DEFAULT, enforce_bounded_range
from apps.dashboard_streamlit.ui.text import INTERVAL_EXPLAIN_80

PAGE_ID = "research_lab"
PAGE_TITLE = "Research Lab"


def render() -> None:
    end = st.date_input(
        "End", value=st.session_state.get("as_of_date", dt.date(2025, 12, 31)), key="ml_end"
    )
    start = st.date_input("Start", value=end - dt.timedelta(days=365), key="ml_start")
    enforce_bounded_range(start, end, DAILY_MAX_DAYS_DEFAULT, page_id=PAGE_ID)
    rows = cached_get_json(
        "/ml/predict",
        params={
            "start": start.strftime("%d-%m-%Y"),
            "end": end.strftime("%d-%m-%Y"),
            "limit": 5000,
            "offset": 0,
        },
        ttl_s=300,
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        latest = str(df["date"].max())
        st.caption(f"Latest predictions date: {latest}")
        st.dataframe(df[df["date"] == latest].head(20), use_container_width=True)

    st.caption(INTERVAL_EXPLAIN_80)
    if st.button("Run walk-forward"):
        st.json(cached_post_json("/ml/backtest", payload={"mode": "walkforward"}, ttl_s=60))
