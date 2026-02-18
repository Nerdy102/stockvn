from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json
from apps.dashboard_streamlit.ui.perf import DAILY_MAX_DAYS_DEFAULT, enforce_bounded_range
from apps.dashboard_streamlit.ui.text import INTERVAL_EXPLAIN_80

PAGE_ID = "research_lab"
PAGE_TITLE = "Research Lab v4"


def _latest_runs() -> list[str]:
    try:
        runs = cached_get_json("/ml/backtests", params={"limit": 20, "offset": 0}, ttl_s=60)
    except Exception:
        return []
    return [str(r.get("run_hash")) for r in runs if isinstance(r, dict) and r.get("run_hash")]


def render() -> None:
    st.title("Research Lab v4")
    end = st.date_input(
        "End", value=st.session_state.get("as_of_date", dt.date(2025, 12, 31)), key="ml_end"
    )
    start = st.date_input("Start", value=end - dt.timedelta(days=365), key="ml_start")
    enforce_bounded_range(start, end, DAILY_MAX_DAYS_DEFAULT, page_id=PAGE_ID)

    if st.button("Run walk-forward"):
        st.json(cached_post_json("/ml/backtest", payload={"mode": "walkforward"}, ttl_s=60))

    try:
        rows = cached_get_json(
            "/ml/predict",
            params={
                "start": start.strftime("%d-%m-%Y"),
                "end": end.strftime("%d-%m-%Y"),
                "limit": 5000,
                "offset": 0,
            },
            ttl_s=300,
        )
    except Exception:
        rows = []
        st.warning("Unable to reach API. Showing empty research lab state.")
    df = pd.DataFrame(rows)
    if not df.empty:
        latest = str(df["date"].max())
        st.caption(f"Latest predictions date: {latest}")
        st.dataframe(df[df["date"] == latest].head(20), use_container_width=True)
    st.caption(INTERVAL_EXPLAIN_80)

    run_hashes = _latest_runs()
    if len(run_hashes) < 2:
        st.info("Need at least 2 runs for compare. Run backtest first.")
        return

    default_a = run_hashes[0]
    default_b = run_hashes[1]
    c1, c2 = st.columns(2)
    run_a = c1.selectbox("Run A", options=run_hashes, index=0)
    run_b = c2.selectbox("Run B", options=run_hashes, index=1 if len(run_hashes) > 1 else 0)

    tabs = st.tabs(["Compare", "Sensitivity", "Stress", "Ablations", "Promotion Checklist"])

    with tabs[0]:
        cmp = cached_get_json("/ml/research_v4/compare", params={"run_a": run_a, "run_b": run_b}, ttl_s=30)
        mdf = pd.DataFrame(cmp.get("metrics", []))
        if not mdf.empty:
            st.dataframe(mdf, use_container_width=True)
            red = mdf[(mdf["metric"] == "MDD") & (mdf["highlight"] == "red")]
            if not red.empty:
                st.error("MDD worsened vs baseline.")

    with tabs[1]:
        sens = cached_get_json("/ml/research_v4/sensitivity", params={"run_hash": run_b}, ttl_s=30)
        heat = pd.DataFrame(sens.get("heatmap", []))
        st.caption("Fixed axes: x=topK, y=rebalance_freq")
        if not heat.empty:
            piv = heat.pivot(index="rebalance_freq", columns="topK", values="sharpe_net")
            st.dataframe(piv, use_container_width=True)
        st.metric("Robustness score", f"{float(sens.get('robustness_score', 0.0)):.4f}")

    with tabs[2]:
        stv = cached_get_json("/ml/research_v4/stress", params={"run_hash": run_b}, ttl_s=30)
        st.json(stv.get("cases", {}))

    with tabs[3]:
        abl = cached_get_json("/ml/research_v4/ablations", params={"run_hash": run_b}, ttl_s=30)
        st.dataframe(pd.DataFrame(abl.get("groups", [])), use_container_width=True)

    with tabs[4]:
        drift_ok = st.checkbox("Drift OK", value=True)
        capacity_ok = st.checkbox("Capacity OK", value=True)
        ck = cached_get_json(
            "/ml/research_v4/promotion_checklist",
            params={"run_hash": run_b, "drift_ok": str(drift_ok).lower(), "capacity_ok": str(capacity_ok).lower()},
            ttl_s=5,
        )
        st.metric("Promotion", "PASS" if ck.get("pass") else "FAIL")
        st.dataframe(pd.DataFrame(ck.get("rules", [])), use_container_width=True)
        if ck.get("fail_reasons"):
            st.warning(f"Fail reasons: {', '.join(ck.get('fail_reasons', []))}")


if __name__ == "__main__":
    render()
