from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import get, post
from apps.dashboard_streamlit.lib.disclaimer import render_global_disclaimer

st.header("Screener & Discovery")
render_global_disclaimer()

col1, col2 = st.columns([2, 1])
with col1:
    screen_path = st.text_input("Screen YAML path", value="configs/screens/demo_screen.yaml")
with col2:
    st.caption("Bạn có thể chỉnh YAML để thêm filters/weights/technical/tags.")

if st.button("Run screen"):
    try:
        data = post("/screeners/run", json={"screen_path": screen_path})
        results = data.get("results", [])
        st.subheader(f"Kết quả: {data.get('screen')}")
        if not results:
            st.warning("Không có kết quả. Hãy chạy worker/seed data trước.")
        else:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.markdown("#### Explainability (Top 1)")
            st.json(results[0].get("explain", {}))
    except Exception as e:
        st.error(f"API error: {e}")

st.markdown("#### Tickers (quick view)")
try:
    tickers = get("/tickers")
    st.dataframe(pd.DataFrame(tickers), use_container_width=True)
except Exception as exc:
    st.warning(f"Không tải được danh sách tickers: {exc}")
