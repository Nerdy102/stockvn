from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard_streamlit.lib.api import get_bytes
from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "raocmoe_lab"
PAGE_TITLE = "RAOCMOE Lab"


def render() -> None:
    st.title("RAOCMOE Lab")
    dataset = st.selectbox("dataset", ["vn_daily", "crypto_15m"], index=1)
    universe = st.text_area("universe", value="BTCUSDT,ETHUSDT")
    c1, c2 = st.columns(2)
    start = c1.date_input("start")
    end = c2.date_input("end")
    if st.button("Run RAOCMOE"):
        payload = {
            "dataset": dataset,
            "universe": [s.strip() for s in universe.split(",") if s.strip()],
            "start": str(start),
            "end": str(end),
        }
        res = cached_post_json("/raocmoe/run", payload, ttl_s=1)
        st.session_state["raocmoe_run_id"] = res["run_id"]

    run_id = st.session_state.get("raocmoe_run_id")
    if run_id:
        run = cached_get_json(f"/lab/runs/{run_id}", None, ttl_s=2)
        st.json(run)
        log = cached_get_json(f"/lab/runs/{run_id}/log", {"tail": 4000}, ttl_s=2)
        st.code(log.get("log", ""))
        if run.get("status") == "SUCCEEDED":
            artifacts = cached_get_json(f"/lab/runs/{run_id}/artifacts", None, ttl_s=2)
            st.dataframe(pd.DataFrame(artifacts), use_container_width=True)
            names = {a["path"] for a in artifacts}
            if "equity.csv" in names:
                raw = get_bytes(f"/lab/runs/{run_id}/artifact", params={"path": "equity.csv"})
                eq = pd.read_csv(io.BytesIO(raw)).head(2000)
                st.plotly_chart(px.line(eq, x="date", y="equity"), use_container_width=True)
            for item in artifacts:
                st.markdown(f"- [{item['path']}](/lab/runs/{run_id}/artifact?path={item['path']})")
