from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "alerts"
PAGE_TITLE = "Alerts"


def render() -> None:
    st.subheader("Create rule")
    name = st.text_input("Rule name", value="my_rule")
    timeframe = st.selectbox("Timeframe", options=["1D"], index=0)
    expr = st.text_input(
        "Expression", value="CROSSOVER(close, SMA(20)) AND volume > 1.5*AVG(volume, 20)"
    )
    symbols = st.text_input("Symbols (comma separated)", value="FVNA,FVNB")

    if st.button("Create rule"):
        payload = {
            "name": name,
            "timeframe": timeframe,
            "expression": expr,
            "symbols": (
                [s.strip() for s in symbols.split(",") if s.strip()] if symbols.strip() else None
            ),
        }
        st.success(cached_post_json("/alerts/rules", payload=payload, ttl_s=60))

    st.subheader("Rules")
    st.dataframe(
        pd.DataFrame(cached_get_json("/alerts/rules", params=None, ttl_s=60)),
        use_container_width=True,
    )
    st.subheader("Events")
    st.dataframe(
        pd.DataFrame(cached_get_json("/alerts/events", params=None, ttl_s=60)),
        use_container_width=True,
    )
