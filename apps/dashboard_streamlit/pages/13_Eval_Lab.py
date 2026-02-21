from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import get_bytes
from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "eval_lab"
PAGE_TITLE = "Eval Lab"


def render() -> None:
    st.title("Eval Lab")
    dataset = st.selectbox("dataset", ["vn_daily", "crypto_15m"])
    if st.button("Run Eval Lab"):
        res = cached_post_json("/eval_lab/run", {"dataset": dataset}, ttl_s=1)
        st.session_state["eval_run_id"] = res["run_id"]

    run_id = st.session_state.get("eval_run_id")
    if run_id:
        run = cached_get_json(f"/lab/runs/{run_id}", None, ttl_s=2)
        st.json(run)
        st.code(
            (cached_get_json(f"/lab/runs/{run_id}/log", {"tail": 4000}, ttl_s=2) or {}).get(
                "log", ""
            )
        )
        if run.get("status") == "SUCCEEDED":
            artifacts = cached_get_json(f"/lab/runs/{run_id}/artifacts", None, ttl_s=2)
            st.dataframe(pd.DataFrame(artifacts), use_container_width=True)
            names = {a["path"] for a in artifacts}
            if "results_table.csv" in names:
                df = pd.read_csv(
                    io.BytesIO(
                        get_bytes(
                            f"/lab/runs/{run_id}/artifact", params={"path": "results_table.csv"}
                        )
                    )
                ).head(2000)
                st.subheader("Scoreboard")
                st.dataframe(df, use_container_width=True)
            if "summary.json" in names:
                obj = pd.read_json(
                    io.BytesIO(
                        get_bytes(f"/lab/runs/{run_id}/artifact", params={"path": "summary.json"})
                    ),
                    typ="series",
                )
                st.subheader("Reliability Verdict")
                st.json(obj.to_dict())
