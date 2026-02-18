from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "new_orders"
PAGE_TITLE = "New Orders"


def render() -> None:
    st.title("New Orders (Paper OMS)")
    portfolios = cached_get_json("/portfolio", params=None, ttl_s=60)
    if not portfolios:
        st.warning("No portfolio available")
        return

    pid = st.selectbox("Portfolio", options=[p["id"] for p in portfolios])
    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.text_input("Symbol", value="AAA")
    side = c2.selectbox("Side", options=["BUY", "SELL"])
    qty = c3.number_input("Qty", min_value=100.0, step=100.0, value=100.0)
    px = c4.number_input("Price", min_value=1.0, step=10.0, value=10000.0)
    client_order_id = st.text_input("Client order id", value=f"paper-{pid}-{symbol}-{side}-{int(qty)}")

    if st.button("Submit paper order"):
        res = cached_post_json(
            "/orders/submit",
            payload={
                "portfolio_id": pid,
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "price": px,
                "adapter": "paper",
            },
            ttl_s=1,
        )
        st.json(res)

    st.subheader("Orders")
    orders = cached_get_json("/orders", params={"portfolio_id": pid, "limit": 200}, ttl_s=2)
    st.dataframe(pd.DataFrame(orders), use_container_width=True)

    st.subheader("Fills")
    fills = cached_get_json("/fills", params={"limit": 200}, ttl_s=2)
    st.dataframe(pd.DataFrame(fills), use_container_width=True)
