from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "run_manager"
PAGE_TITLE = "Run Manager"


def render() -> None:
    st.title("Run Manager")
    c1, c2 = st.columns(2)
    run_type = c1.selectbox(
        "run_type", ["", "RAOCMOE_BACKTEST", "EVAL_LAB", "DATA_INGEST", "SEED_DB"]
    )
    status = c2.selectbox("status", ["", "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"])
    params = {"limit": 100}
    if run_type:
        params["run_type"] = run_type
    if status:
        params["status"] = status
    runs = cached_get_json("/lab/runs", params, ttl_s=2)
    df = pd.DataFrame(runs)
    st.dataframe(df, use_container_width=True)
    run_id = st.text_input("run_id") or st.session_state.get("selected_run_id", "")
    if run_id:
        st.json(cached_get_json(f"/lab/runs/{run_id}", None, ttl_s=2))
        st.code(
            (cached_get_json(f"/lab/runs/{run_id}/log", {"tail": 4000}, ttl_s=2) or {}).get(
                "log", ""
            )
        )
        st.dataframe(
            pd.DataFrame(cached_get_json(f"/lab/runs/{run_id}/artifacts", None, ttl_s=2)),
            use_container_width=True,
        )
        if st.button("Cancel run"):
            st.json(cached_post_json(f"/lab/runs/{run_id}/cancel", {}, ttl_s=1))
