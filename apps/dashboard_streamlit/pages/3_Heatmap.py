from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard_streamlit.lib.api import get
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("Heatmap: Sector / Top Movers / Breadth / Correlation")
render_global_disclaimer()


@st.cache_data(ttl=3600)
def _tickers_universe() -> list[dict]:
    return get("/tickers", params={"limit": 2000, "offset": 0})


@st.cache_data(ttl=900)
def _prices_last_365(symbol: str) -> list[dict]:
    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    return get(
        "/prices",
        params={
            "symbol": symbol,
            "timeframe": "1D",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 2000,
            "offset": 0,
        },
    )


tickers = _tickers_universe()
df_t = pd.DataFrame(tickers)
symbols = df_t["symbol"].tolist()

rows = []
for sym in symbols:
    try:
        rows.extend(_prices_last_365(sym))
    except Exception:
        continue

df_p = pd.DataFrame(rows)
if df_p.empty:
    st.warning("No price data.")
    st.stop()

df_p["timestamp"] = pd.to_datetime(df_p["timestamp"])
df_p["date"] = df_p["timestamp"].dt.date

panel = df_p.pivot(index="date", columns="symbol", values="close").sort_index()
ret = panel.pct_change()

last = ret.iloc[-1].dropna()
adv = int((last > 0).sum())
dec = int((last < 0).sum())
st.metric("Breadth (last day)", value=f"Adv {adv} / Dec {dec}")

last_5d = ret.tail(5).mean().dropna()
df_map = df_t.set_index("symbol")[["sector"]].join(last_5d.rename("ret"), how="inner")
sector_ret = df_map.groupby("sector")["ret"].mean().sort_values(ascending=False)

st.subheader("Sector performance (avg return ~5D)")
fig1 = px.imshow(sector_ret.to_frame().T, aspect="auto")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Top movers (avg return ~5D)")
movers = df_map.sort_values("ret", ascending=False).head(10)
st.dataframe(movers.reset_index(), use_container_width=True)

st.subheader("Correlation (returns, ~120D)")
corr = ret.drop(columns=["VNINDEX"], errors="ignore").tail(120).corr()
fig2 = px.imshow(corr, aspect="auto")
st.plotly_chart(fig2, use_container_width=True)

st.caption("Demo universe nhỏ => bulk load bằng loop (MVP). Khi production nên làm endpoint bulk.")
