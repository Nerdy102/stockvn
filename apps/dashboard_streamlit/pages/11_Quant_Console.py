from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "quant_console"
PAGE_TITLE = "Quant Console"


def _safe_get(path: str, params=None, ttl_s: int = 2):
    try:
        return cached_get_json(path, params=params, ttl_s=ttl_s)
    except Exception as exc:
        st.warning(f"API unavailable for {path}: {exc}")
        return None


def render() -> None:
    st.title("Interactive Quant Console")
    health = _safe_get("/health", ttl_s=2) or {}
    gov = _safe_get("/governance/status", ttl_s=2) or {}
    realtime = _safe_get("/realtime/summary", ttl_s=2) or {}
    st.subheader("System Health")
    st.json({"health": health, "governance": gov, "realtime": realtime})

    c1, c2, c3 = st.columns(3)
    if c1.button("Run RAOCMOE Backtest (default)"):
        res = cached_post_json(
            "/raocmoe/run",
            {
                "dataset": "crypto_15m",
                "universe": ["BTCUSDT", "ETHUSDT"],
                "start": "2023-01-01",
                "end": "2023-12-31",
            },
            ttl_s=1,
        )
        st.success(f"Started {res.get('run_id')}")
    if c2.button("Run Eval Lab (default)"):
        res = cached_post_json("/eval_lab/run", {"dataset": "vn_daily"}, ttl_s=1)
        st.success(f"Started {res.get('run_id')}")
    if c3.button("Open latest SUCCEEDED run"):
        runs = _safe_get("/lab/runs", params={"status": "SUCCEEDED", "limit": 1}, ttl_s=1) or []
        if runs:
            st.session_state["selected_run_id"] = runs[0]["run_id"]
            st.info(f"Selected {runs[0]['run_id']}")

    st.subheader("Latest Runs")
    runs = _safe_get("/lab/runs", params={"limit": 20}, ttl_s=2) or []
    st.dataframe(pd.DataFrame(runs), use_container_width=True)
