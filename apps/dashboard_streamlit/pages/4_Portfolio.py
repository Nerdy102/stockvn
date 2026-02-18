from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "portfolio"
PAGE_TITLE = "Portfolio Command Center v4"


def _show_constraints_panel(constraints: dict) -> None:
    st.subheader("Constraints")
    active = constraints.get("active", [])
    if active:
        st.caption("Binding/active constraints")
        st.write(", ".join(str(x) for x in active))
    c1, c2 = st.columns(2)
    c1.json({"violations_pre": constraints.get("violations_pre", [])})
    c2.json({"violations_post": constraints.get("violations_post", [])})
    st.json({"distance_metrics": constraints.get("distance_metrics", {})})


def render() -> None:
    portfolios = cached_get_json("/portfolio", params=None, ttl_s=300)
    if not portfolios and st.button("Create demo portfolio"):
        cached_post_json("/portfolio", payload={"name": "Demo Portfolio"}, ttl_s=60)
        portfolios = cached_get_json("/portfolio", params=None, ttl_s=300)

    if not portfolios:
        st.warning("Chưa có portfolio.")
        return

    pid = st.selectbox("Portfolio", options=[p["id"] for p in portfolios], format_func=lambda x: f"#{x}")
    dashboard = cached_get_json("/portfolio/dashboard", params={"portfolio_id": pid}, ttl_s=120)
    preview = cached_get_json(f"/portfolio/{pid}/rebalance-preview", params=None, ttl_s=60)
    scenarios = cached_get_json(f"/portfolio/{pid}/scenario-lab", params=None, ttl_s=60)

    st.title("Portfolio Command Center v4")
    gov = cached_get_json("/governance/status", params=None, ttl_s=5)
    g1, g2, g3 = st.columns(3)
    g1.metric("Governance", str(gov.get("status", "RUNNING")))
    g2.metric("Pause reason", str(gov.get("pause_reason", "") or "-"))
    last_recon = gov.get("last_reconciliation") or {}
    g3.metric("Last reconcile", str(last_recon.get("status", "n/a")))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NAV", f"{dashboard.get('nav', 0):,.0f}")
    c2.metric("Cash", f"{dashboard.get('cash', 0):,.0f}")
    risk = dashboard.get("risk", {})
    c3.metric("Vol annual", f"{risk.get('vol', 0):.2%}")
    c4.metric("CVaR 5%", f"{risk.get('cvar', 0):.2%}")

    st.subheader("Holdings")
    h = pd.DataFrame(dashboard.get("holdings", []))
    if not h.empty:
        sectors = ["ALL"] + sorted(h["sector"].dropna().unique().tolist())
        sec = st.selectbox("Sector filter", sectors)
        min_w = st.slider("Min weight", 0.0, 0.3, 0.0, 0.01)
        if sec != "ALL":
            h = h[h["sector"] == sec]
        h = h[h["weight"] >= min_w]
        st.dataframe(h, use_container_width=True)

    st.subheader("Risk panel")
    r1, r2, r3 = st.columns(3)
    r1.metric("Beta", f"{risk.get('beta', 0):.3f}")
    r2.metric("MDD", f"{risk.get('mdd', 0):.2%}")
    r3.metric("VaR hist", f"{risk.get('var_hist', 0):.2%}")

    rc = dashboard.get("risk_contrib", {})
    st.write("Risk contribution by names")
    st.dataframe(pd.DataFrame(rc.get("names", [])), use_container_width=True)

    _show_constraints_panel(dashboard.get("constraints", {}))

    st.subheader("Capacity")
    cap = dashboard.get("capacity", {})
    st.dataframe(pd.DataFrame(cap.get("by_symbol", [])), use_container_width=True)
    if cap.get("flags"):
        st.error(f"Capacity flags: {len(cap.get('flags', []))}")

    st.subheader("Rebalance preview")
    st.dataframe(pd.DataFrame(preview.get("trades", [])), use_container_width=True)
    st.json({"expected_costs": preview.get("expected_costs", {})})
    st.dataframe(pd.DataFrame(preview.get("ac_schedule", [])), use_container_width=True)
    with st.expander("Explain reason keys"):
        st.write(preview.get("explain_reason_keys", []))

    st.subheader("Scenario lab (S1..S4)")
    st.json(scenarios.get("scenarios", {}))


if __name__ == "__main__":
    render()
