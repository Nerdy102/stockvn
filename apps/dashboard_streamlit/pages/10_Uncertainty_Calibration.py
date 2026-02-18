from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json
from apps.dashboard_streamlit.ui.perf import DAILY_MAX_DAYS_DEFAULT, enforce_bounded_range


PAGE_ID = "uncertainty_calibration"


def render() -> None:
    st.title("Uncertainty & Calibration Terminal")
    end = st.date_input("End", value=st.session_state.get("as_of_date", dt.date.today()), key="uncert_end")
    start = st.date_input("Start", value=end - dt.timedelta(days=365), key="uncert_start")
    enforce_bounded_range(start, end, DAILY_MAX_DAYS_DEFAULT, page_id=PAGE_ID)

    try:
        body = cached_get_json(
            "/ml/alpha_v3_cp/uncertainty_terminal",
            params={"end": end.strftime("%d-%m-%Y"), "window": 252},
            ttl_s=120,
        )
    except Exception:
        st.warning("Unable to reach API. Showing empty terminal state.")
        body = {"calibration_metrics": [], "prob_calibration_metrics": {}, "reset_events": [], "alerts": {}}

    alerts = body.get("alerts", {})
    if alerts.get("critical_undercoverage"):
        st.error("CRITICAL: interval coverage below 0.75. Model is currently overconfident.")
    if alerts.get("warn_ece"):
        st.warning("WARN: ECE above 0.05. Probability calibration drift detected.")

    cal = pd.DataFrame(body.get("calibration_metrics", []))
    overall = cal[cal["group_key"] == "ALL"] if not cal.empty else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    cov = float(overall["coverage"].iloc[0]) if not overall.empty else 0.0
    sharp = float(overall["sharpness_median"].iloc[0]) if not overall.empty else 0.0
    ece = float((body.get("prob_calibration_metrics") or {}).get("ece", 0.0))
    c1.metric("Coverage", f"{cov:.2%}")
    c2.metric("Sharpness median width", f"{sharp:.4f}")
    c3.metric("ECE", f"{ece:.4f}")

    if not cal.empty:
        st.subheader("Coverage sparkline (overall)")
        s = cal[cal["group_key"] == "ALL"][["date_end", "coverage"]].copy()
        if not s.empty:
            s = s.sort_values("date_end")
            st.line_chart(s.set_index("date_end")["coverage"])

        st.subheader("By liquidity bucket and regime")
        grp = cal[cal["group_key"].str.contains("bucket:")].copy()
        st.dataframe(grp[["group_key", "coverage", "gap", "sharpness_median", "width_p90", "count"]], use_container_width=True)

        st.subheader("Width distribution summary")
        wd = overall[["sharpness_median", "width_p90"]].copy() if not overall.empty else pd.DataFrame()
        if not wd.empty:
            st.bar_chart(wd.T.rename(columns={wd.index[0]: "value"}))

    st.subheader("Reliability diagram")
    rel = pd.DataFrame((body.get("prob_calibration_metrics") or {}).get("reliability_bins_json", []))
    if not rel.empty:
        chart_df = rel[["bin", "avg_pred", "freq"]].copy().set_index("bin")
        st.line_chart(chart_df)
        st.dataframe(rel, use_container_width=True)

    st.markdown("---")
    st.subheader("ListNet v2 Calibrated Outperformance Probability")
    try:
        rel_v2 = cached_get_json(
            "/ml/listnet_v2/reliability",
            params={"end": end.strftime("%d-%m-%Y"), "window": 252},
            ttl_s=120,
        )
    except Exception:
        rel_v2 = {"prob_calibration_metrics": {}, "governance_warning": False, "rolling_ece_20": []}

    pmet = rel_v2.get("prob_calibration_metrics") or {}
    v2_brier = float(pmet.get("brier", 0.0))
    v2_ece = float(pmet.get("ece", 0.0))
    k1, k2 = st.columns(2)
    k1.metric("ListNet Brier", f"{v2_brier:.4f}")
    k2.metric("ListNet ECE(10)", f"{v2_ece:.4f}")

    if bool(rel_v2.get("governance_warning", False)):
        st.warning("Governance Warning: rolling 20-trading-day ECE > 0.05 for ListNet v2")

    rel2 = pd.DataFrame(pmet.get("reliability_bins_json", []))
    if not rel2.empty:
        st.line_chart(rel2[["bin", "avg_pred", "freq"]].set_index("bin"))

    rolling = pd.DataFrame(rel_v2.get("rolling_ece_20", []))
    if not rolling.empty:
        st.line_chart(rolling.set_index("date")[["ece"]])

    st.subheader("Reset events timeline")
    events = pd.DataFrame(body.get("reset_events", []))
    if events.empty:
        st.caption("No reset events in selected window.")
    else:
        events = events.sort_values("date")
        st.dataframe(events[["date", "event_type", "before_coverage", "after_coverage"]], use_container_width=True)


if __name__ == "__main__":
    render()
