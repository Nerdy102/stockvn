from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import post
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("Alpha v2 Lab")
render_global_disclaimer()
st.caption("NET backtest includes fee + tax + slippage + fill penalty. past ≠ future, overfit risk, liquidity/limit risk.")

if st.button("Run diagnostics v2"):
    d = post("/ml/diagnostics", json={})
    st.subheader("Diagnostics tables")
    st.json(d)

if st.button("Run backtest v2"):
    b = post("/ml/backtest", json={"mode": "v2"})
    wf = b.get("walk_forward", {})
    curve = pd.DataFrame(wf.get("equity_curve", []))
    if not curve.empty and "equity" in curve:
        st.line_chart(curve.set_index("as_of_date")["equity"] if "as_of_date" in curve else curve["equity"])
    st.subheader("IC decay")
    diag = post("/ml/diagnostics", json={})
    decay = {k: v for k, v in diag.get("metrics", {}).items() if k.startswith("ic_decay_")}
    st.table(pd.DataFrame([decay]))
    st.subheader("Bootstrap CI")
    ci = {k: v for k, v in diag.get("metrics", {}).items() if k.endswith("_lo") or k.endswith("_hi")}
    st.table(pd.DataFrame([ci]))
    st.subheader("Regime breakdown")
    reg = {k: v for k, v in diag.get("metrics", {}).items() if "trend_up" in k or "sideways" in k or "risk_off" in k}
    st.table(pd.DataFrame([reg]))
