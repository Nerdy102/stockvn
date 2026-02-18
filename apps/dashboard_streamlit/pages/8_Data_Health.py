from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json

PAGE_ID = "data_health"
PAGE_TITLE = "Data Health v3"


def render() -> None:
    st.title("Data Health v3")
    try:
        summary = cached_get_json("/data/health/summary", params=None, ttl_s=30)
    except Exception:
        st.warning("Unable to reach data health API.")
        summary = {"open_incidents": [], "sla_gauges": {}, "counts": {}}

    gauges = summary.get("sla_gauges", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Open incidents", int(gauges.get("open_count", 0)))
    c2.metric("SLA breach >24h", int(gauges.get("breach_24h", 0)))
    c3.metric("SLA breach >72h", int(gauges.get("breach_72h", 0)))

    st.subheader("Open incidents")
    open_df = pd.DataFrame(summary.get("open_incidents", []))
    st.dataframe(open_df, use_container_width=True)

    st.subheader("Realtime Ops")
    try:
        rt_ops = cached_get_json("/data/health/realtime_ops", params=None, ttl_s=10)
    except Exception:
        rt_ops = {"gauges": {}, "incidents": [], "runbooks": {}, "snapshots": []}

    gauges_rt = rt_ops.get("gauges", {})
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Ingest lag p95 (s)", f"{float(gauges_rt.get('ingest_lag_s_p95', 0.0)):.2f}")
    r2.metric("Bar build p95 (s)", f"{float(gauges_rt.get('bar_build_latency_s_p95', 0.0)):.2f}")
    r3.metric("Signal latency p95 (s)", f"{float(gauges_rt.get('signal_latency_s_p95', 0.0)):.2f}")
    r4.metric("Redis pending", f"{float(gauges_rt.get('redis_stream_pending', 0.0)):,.0f}")

    st.caption("Runbook IDs")
    st.json(rt_ops.get("runbooks", {}))
    st.caption("Realtime incidents")
    st.dataframe(pd.DataFrame(rt_ops.get("incidents", [])), use_container_width=True)

    st.subheader("Incident detail")
    try:
        incidents = cached_get_json("/data/health/incidents", params={"limit": 200}, ttl_s=30)
    except Exception:
        incidents = []
    if not incidents:
        st.caption("No incidents")
        return

    ids = [int(r["id"]) for r in incidents]
    picked = st.selectbox("Incident ID", options=ids)
    detail = cached_get_json(f"/data/health/incidents/{picked}", params=None, ttl_s=10)

    st.json({
        "summary": detail.get("summary"),
        "source": detail.get("source"),
        "severity": detail.get("severity"),
        "status": detail.get("status"),
        "runbook_section": detail.get("runbook_section"),
    })
    st.write("Suggested actions")
    st.json(detail.get("suggested_actions_json", {}))
    st.write("Details")
    st.json(detail.get("details_json", {}))


if __name__ == "__main__":
    render()
