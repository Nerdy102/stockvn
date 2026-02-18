from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.indicators import add_indicators
from core.technical import auto_trendline, detect_supply_demand_zones

from apps.dashboard_streamlit.lib.api import get
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("Charting & Signals")
render_global_disclaimer()


@st.cache_data(ttl=3600)
def _tickers_universe() -> list[dict]:
    return get("/tickers", params={"limit": 2000, "offset": 0})


@st.cache_data(ttl=900)
def _prices_last_365(symbol: str, timeframe: str) -> list[dict]:
    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    return get(
        "/prices",
        params={
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 2000,
            "offset": 0,
        },
    )


tickers = []
try:
    tickers = _tickers_universe()
except Exception:
    tickers = []

symbols = [t["symbol"] for t in tickers] if tickers else ["FVNA", "FVNB", "VNINDEX"]
symbol = st.selectbox("Symbol", options=symbols, index=0)
timeframe = st.selectbox("Timeframe", options=["1D", "1W", "60m", "15m"], index=0)

show_sma = st.checkbox("Overlay SMA20", value=True)
show_ema = st.checkbox("Overlay EMA20", value=True)
show_zones = st.checkbox("Auto supply/demand zones (MVP)", value=True)
show_trendline = st.checkbox("Auto trendline (MVP)", value=True)

if st.button("Load chart"):
    tf = "1D" if timeframe == "1W" else timeframe
    prices = _prices_last_365(symbol, tf)
    if not prices:
        st.warning("No price data.")
        st.stop()

    df = pd.DataFrame(prices)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    ohlcv = df[["timestamp", "open", "high", "low", "close", "volume"]].copy().set_index("timestamp")

    if timeframe == "1W":
        ohlcv = (
            ohlcv.resample("W-FRI")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

    ind = add_indicators(ohlcv)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                close=ohlcv["close"],
                name="Price",
            )
        ]
    )
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], name="Volume", yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"))

    if show_sma:
        fig.add_trace(go.Scatter(x=ind.index, y=ind["SMA20"], mode="lines", name="SMA20"))
    if show_ema:
        fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA20"], mode="lines", name="EMA20"))

    if show_zones:
        zones = detect_supply_demand_zones(ohlcv)
        if zones:
            zdf = pd.DataFrame(
                [
                    {"kind": z.kind, "start": z.start, "end": z.end, "low": z.low, "high": z.high}
                    for z in zones
                ]
            )
            st.markdown("#### Zones (edit để chỉnh tay, MVP)")
            zedit = st.data_editor(zdf, use_container_width=True, num_rows="dynamic")
            for _, z in zedit.iterrows():
                fig.add_shape(
                    type="rect",
                    x0=pd.to_datetime(z["start"]),
                    x1=pd.to_datetime(z["end"]),
                    y0=float(z["low"]),
                    y1=float(z["high"]),
                    opacity=0.15,
                    line_width=1,
                )

    if show_trendline and len(ohlcv) >= 20:
        tl = auto_trendline(ohlcv, kind="support", lookback=min(120, len(ohlcv)))
        y = [tl["slope"] * i + tl["intercept"] for i in range(len(ohlcv))]
        fig.add_trace(go.Scatter(x=ohlcv.index, y=y, mode="lines", name="Trendline (support)"))

    st.plotly_chart(fig, use_container_width=True)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=ind.index, y=ind["RSI14"], mode="lines", name="RSI14"))
    fig_rsi.add_hline(y=70)
    fig_rsi.add_hline(y=30)
    fig_rsi.update_layout(height=240, title="RSI14")
    st.plotly_chart(fig_rsi, use_container_width=True)

st.caption("MVP: W resample từ 1D. Zones/Trendline: heuristic + chỉnh tay qua data_editor.")
