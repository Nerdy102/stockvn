from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "realtime_console"
PAGE_TITLE = "Realtime Console"


def render() -> None:
    st.title("Realtime Console")
    c1, c2, c3 = st.columns(3)
    symbol = c1.text_input("symbol", value="BTCUSDT")
    tf = c2.selectbox("tf", ["1m", "15m", "60m", "1D"], index=1)
    limit = c3.slider("limit", 50, 2000, 300)
    refresh = st.slider("refresh_s", 1, 10, 2)
    st.caption(f"Polling every {refresh}s")

    st.json(cached_get_json("/realtime/summary", None, ttl_s=refresh))
    bars = cached_get_json(
        "/prices", {"symbol": symbol, "timeframe": tf, "limit": limit}, ttl_s=refresh
    )
    df = pd.DataFrame(bars).head(2000)
    st.dataframe(df.tail(200), use_container_width=True)
    if not df.empty and {"open", "high", "low", "close", "timestamp"}.issubset(df.columns):
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("One-click paper order")
    pid = st.number_input("portfolio_id", min_value=1, value=1)
    side = st.selectbox("side", ["BUY", "SELL"])
    qty = st.number_input("qty", min_value=1.0, value=100.0)
    px = float(df["close"].iloc[-1]) if not df.empty and "close" in df.columns else 10000.0
    st.write(f"Auto price: {px:.2f}")
    if st.button("Submit paper order"):
        st.json(
            cached_post_json(
                "/orders/submit",
                {
                    "portfolio_id": int(pid),
                    "client_order_id": f"rt-{symbol}-{side}-{int(qty)}",
                    "symbol": symbol,
                    "side": side,
                    "quantity": float(qty),
                    "price": px,
                    "adapter": "paper",
                },
                ttl_s=1,
            )
        )
    st.dataframe(
        pd.DataFrame(cached_get_json("/orders", {"portfolio_id": int(pid), "limit": 100}, ttl_s=2)),
        use_container_width=True,
    )
    st.dataframe(
        pd.DataFrame(cached_get_json("/fills", {"limit": 100}, ttl_s=2)), use_container_width=True
    )
