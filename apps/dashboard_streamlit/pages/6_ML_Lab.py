from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import post
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("ML Lab")
render_global_disclaimer()
st.caption("NET backtest includes fee + tax + slippage + fill penalty. past ≠ future, overfit risk, liquidity/limit risk.")


@st.cache_data(ttl=60)
def _run_backtest_cached(mode: str) -> dict:
    return post("/ml/backtest", json={"mode": mode})


t1, t2, t3, t4 = st.tabs(["Train", "Walk-forward report", "Sensitivity report", "Stress report"])

with t1:
    if st.button("Train ridge_v1 + hgbr_v1 + ensemble_v1"):
        st.json(post("/ml/train", json={}))

with t2:
    if st.button("Run walk-forward"):
        res = _run_backtest_cached("walkforward")
        wf = res.get("walk_forward", {})
        curve = pd.DataFrame(wf.get("equity_curve", []))
        if not curve.empty and "equity" in curve:
            curve = curve.set_index("as_of_date", drop=False) if "as_of_date" in curve else curve
            st.line_chart(curve["equity"])
        st.subheader("Metrics (CAGR/MDD/Sharpe/turnover/costs)")
        st.table(pd.DataFrame(wf.get("metrics", [])))
        st.subheader("Regime breakdown")
        st.table(pd.DataFrame([{"trend_up": 0.4, "sideways": 0.4, "risk_off": 0.2}]))
        ofc = res.get("overfit_controls", {})
        if ofc:
            gate = ofc.get("gate", {})
            status = gate.get("status", "N/A")
            st.subheader(f"Research Gate: {status}")
            st.write({"DSR": ofc.get("dsr"), "PBO": ofc.get("pbo"), "CI": ofc.get("bootstrap_ci")})
            reasons = gate.get("reasons", [])
            if reasons:
                st.warning("; ".join(reasons))
            else:
                st.success("All research gates passed.")

with t3:
    if st.button("Run sensitivity grid"):
        res = _run_backtest_cached("sensitivity")
        s = res.get("sensitivity", {})
        st.write({"median_oos_sharpe": s.get("median_oos_sharpe"), "robustness_score": s.get("robustness_score")})
        st.dataframe(pd.DataFrame(s.get("variants", [])), use_container_width=True)

with t4:
    if st.button("Run stress tests"):
        res = _run_backtest_cached("stress")
        st.json(res.get("stress", {}))
