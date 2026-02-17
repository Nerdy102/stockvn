from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import get, post
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("Alerts (DSL)")
render_global_disclaimer()

st.subheader("Create rule")
name = st.text_input("Rule name", value="my_rule")
timeframe = st.selectbox("Timeframe", options=["1D"], index=0)
expr = st.text_input(
    "Expression", value="CROSSOVER(close, SMA(20)) AND volume > 1.5*AVG(volume, 20)"
)
symbols = st.text_input("Symbols (comma separated, empty=all)", value="FVNA,FVNB")

if st.button("Create rule"):
    payload = {
        "name": name,
        "timeframe": timeframe,
        "expression": expr,
        "symbols": (
            [s.strip() for s in symbols.split(",") if s.strip()] if symbols.strip() else None
        ),
    }
    r = post("/alerts/rules", json=payload)
    st.success(r)

st.subheader("Rules")
rules = get("/alerts/rules")
st.dataframe(pd.DataFrame(rules), use_container_width=True)

st.subheader("Events")
events = get("/alerts/events")
st.dataframe(pd.DataFrame(events), use_container_width=True)

st.caption("Worker sẽ chạy theo interval và đánh giá rules để sinh events (MVP).")
