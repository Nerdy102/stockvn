from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import api_base
from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "data_manager"
PAGE_TITLE = "Data Manager"


def render() -> None:
    st.title("Data Manager")
    up = st.file_uploader("Upload prices CSV", type=["csv"])
    if up is not None and st.button("Upload"):
        with httpx.Client(timeout=60) as client:
            r = client.post(
                api_base() + "/data/upload", files={"file": (up.name, up.getvalue(), "text/csv")}
            )
            r.raise_for_status()
            st.json(r.json())

    c1, c2 = st.columns(2)
    if c1.button("Ingest data_drop"):
        st.json(cached_post_json("/data/ingest", {}, ttl_s=1))
    if c2.button("Seed DB"):
        st.json(cached_post_json("/data/seed", {}, ttl_s=1))

    st.subheader("Audit latest")
    st.json(cached_get_json("/data/audit/latest", None, ttl_s=5))
    tickers = cached_get_json("/tickers", {"limit": 500}, ttl_s=60)
    st.subheader("Universe Builder")
    st.dataframe(pd.DataFrame(tickers).head(2000), use_container_width=True)
