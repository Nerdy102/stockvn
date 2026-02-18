from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "alerts"
PAGE_TITLE = "Alerts Triage Board v5"


def _render_card(row: dict) -> None:
    st.markdown(f"**#{row['id']} {row['symbol']}**")
    st.caption(
        f"date={row['date']} | sev={row['severity']} | SLA={row['sla_timer_trading_days']} TD"
    )
    st.json(row.get("reason_json", {}))


def render() -> None:
    st.title("Alerts Triage Board v5")
    try:
        board = cached_get_json("/alerts/v5", params=None, ttl_s=30)
    except Exception:
        st.warning("Unable to reach API. Showing empty board.")
        board = {"states": {"NEW": [], "ACK": [], "RESOLVED": []}}
    states = board.get("states", {"NEW": [], "ACK": [], "RESOLVED": []})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("NEW")
        for r in states.get("NEW", []):
            with st.container(border=True):
                _render_card(r)
    with c2:
        st.subheader("ACK")
        for r in states.get("ACK", []):
            with st.container(border=True):
                _render_card(r)
    with c3:
        st.subheader("RESOLVED")
        for r in states.get("RESOLVED", []):
            with st.container(border=True):
                _render_card(r)

    st.subheader("Actions")
    all_rows = states.get("NEW", []) + states.get("ACK", []) + states.get("RESOLVED", [])
    if all_rows:
        alert_id = st.selectbox("Alert ID", options=[r["id"] for r in all_rows])
        action = st.selectbox("Action", options=["ACK", "RESOLVE", "SNOOZE"])
        snooze_until = st.date_input("Snooze until", value=dt.date.today() + dt.timedelta(days=1))
        if st.button("Apply action"):
            payload = {"action": action}
            if action == "SNOOZE":
                payload["snooze_until"] = snooze_until.strftime("%Y-%m-%d")
            out = cached_post_json(f"/alerts/v5/{alert_id}/action", payload=payload, ttl_s=5)
            st.success(out)

    st.subheader("Digest log")
    try:
        logs = cached_get_json("/alerts/v5/digest-log", params=None, ttl_s=30)
    except Exception:
        logs = []
    st.dataframe(pd.DataFrame(logs), use_container_width=True)


if __name__ == "__main__":
    render()
