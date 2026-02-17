from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="VN Invest Toolkit", layout="wide")

st.title("VN Invest Toolkit (Offline Demo)")
st.caption("⚠️ Công cụ học tập/giáo dục. Không phải lời khuyên đầu tư. Demo data là synthetic.")

st.markdown(
    """
Dùng sidebar để chuyển trang:

- **Screener**: filters + factor ranking + setup kỹ thuật
- **Charting**: candlestick + indicators + signals overlay + multi timeframe
- **Heatmap**: ngành, top movers, breadth, correlation
- **Portfolio**: import trades + P&L + risk + attribution + rebalance (MVP)
- **Alerts**: rule builder (DSL) + events
"""
)

st.info("Gợi ý: chạy `make run-api` và `make run-worker` trước khi mở dashboard.")
