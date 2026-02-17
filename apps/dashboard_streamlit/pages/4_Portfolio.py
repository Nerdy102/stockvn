from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import get, post
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("Portfolio Tracker")
render_global_disclaimer()

portfolios = get("/portfolio")
if not portfolios:
    if st.button("Create demo portfolio"):
        post("/portfolio", json={"name": "Demo Portfolio"})
    portfolios = get("/portfolio")

if not portfolios:
    st.warning("Chưa có portfolio.")
    st.stop()

pid = st.selectbox(
    "Portfolio", options=[p["id"] for p in portfolios], format_func=lambda x: f"#{x}"
)

st.subheader("Import trades CSV")
st.caption("CSV columns: trade_date,symbol,side,quantity,price,strategy_tag,notes")

uploaded = st.file_uploader("Upload trades CSV", type=["csv"])
if uploaded is not None and st.button("Import uploaded CSV"):
    df = pd.read_csv(uploaded)
    trades = df.to_dict(orient="records")
    res = post(f"/portfolio/{pid}/trades/import", json=trades)
    st.success(res)

if st.button("Import bundled demo trades (data_demo/trades_demo.csv)"):
    df = pd.read_csv("data_demo/trades_demo.csv")
    trades = df.to_dict(orient="records")
    res = post(f"/portfolio/{pid}/trades/import", json=trades)
    st.success(res)

st.subheader("Add trade (manual)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    trade_date = st.text_input("trade_date (YYYY-MM-DD)", value="2026-02-13")
with col2:
    symbol = st.text_input("symbol", value="FVNA")
with col3:
    side = st.selectbox("side", options=["BUY", "SELL"], index=0)
with col4:
    quantity = st.number_input("quantity", min_value=0.0, value=1000.0, step=100.0)
with col5:
    price = st.number_input("price", min_value=0.0, value=40000.0, step=50.0)
strategy_tag = st.text_input("strategy_tag", value="manual")
notes = st.text_input("notes", value="")

if st.button("Submit manual trade"):
    trades = [
        {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "strategy_tag": strategy_tag,
            "notes": notes,
        }
    ]
    res = post(f"/portfolio/{pid}/trades/import", json=trades)
    st.success(res)

if st.button("Refresh summary"):
    s = get(f"/portfolio/{pid}/summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cash now (inferred)", value=f"{s.get('cash_now', 0):,.0f} VND")
    c2.metric("TWR (MVP)", value=f"{s.get('twr', 0)*100:.2f}%")
    risk = s.get("risk", {}) or {}
    c3.metric("Max Drawdown", value=f"{risk.get('max_drawdown', 0)*100:.2f}%")

    st.subheader("Positions")
    st.dataframe(pd.DataFrame(s.get("positions", [])), use_container_width=True)

    st.subheader("Realized trades (fixed SELL clamp for fee/tax correctness)")
    st.dataframe(pd.DataFrame(s.get("realized_trades", [])), use_container_width=True)

    st.subheader("Realized breakdown")
    rb = s.get("realized_breakdown", {}) or {}
    st.write("By day")
    st.dataframe(pd.DataFrame(rb.get("by_day", [])), use_container_width=True)
    st.write("By strategy_tag")
    st.dataframe(pd.DataFrame(rb.get("by_strategy", [])), use_container_width=True)

    st.subheader("Unrealized P&L by sector (MVP)")
    st.json(s.get("unrealized_by_sector", {}))

    st.subheader("Exposure by sector")
    st.json(s.get("exposure_by_sector", {}))

    st.subheader("Concentration")
    st.json(s.get("concentration", {}))

    st.subheader("Risk & Return (MVP)")
    st.json({"risk": s.get("risk"), "twr": s.get("twr")})

    st.subheader("Assumptions")
    st.json(s.get("assumptions", {}))

    st.subheader("Correlation matrix (holdings)")
    corr = s.get("correlation_matrix", {})
    if corr:
        st.dataframe(pd.DataFrame(corr), use_container_width=True)
    else:
        st.info("Not enough holdings/returns to compute correlation.")

    st.subheader("Attribution (MVP)")
    st.json(s.get("attribution_mvp", {}))

    st.subheader("Rebalance suggestion (MVP)")
    st.json(s.get("rebalance_mvp", {}))

st.caption(
    "Cost basis: Average Cost. Fees/taxes theo configs. Demo cash được suy luận để không âm (MVP)."
)
