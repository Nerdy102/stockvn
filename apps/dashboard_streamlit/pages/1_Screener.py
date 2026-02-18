from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "screener"
PAGE_TITLE = "Screener"


def _default_screen() -> dict:
    return {
        "name": "screen-v4",
        "as_of_date": st.session_state.get("as_of_date", dt.date(2025, 12, 31)).isoformat(),
        "universe": {"preset": "ALL"},
        "filters": {
            "min_adv20_value": 1_000_000_000.0,
            "sector_in": [],
            "exchange_in": [],
            "tags_any": [],
            "tags_all": [],
            "neutralization": {"enabled": False},
        },
        "factor_weights": {
            "value": 0.2,
            "quality": 0.2,
            "momentum": 0.2,
            "lowvol": 0.2,
            "dividend": 0.2,
        },
        "technical_setups": {
            "breakout": True,
            "trend": True,
            "pullback": False,
            "volume_spike": False,
        },
    }


def render() -> None:
    st.subheader("Screener UX v4")

    screens_resp = cached_get_json("/screeners", params=None, ttl_s=60)
    saved_screens = screens_resp.get("screens", []) if isinstance(screens_resp, dict) else []

    if "screen_builder" not in st.session_state:
        st.session_state["screen_builder"] = _default_screen()

    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### Builder")
        screen_name = st.text_input(
            "Name", value=st.session_state["screen_builder"].get("name", "screen-v4")
        )
        as_of_date = st.date_input(
            "As of date",
            value=dt.date.fromisoformat(st.session_state["screen_builder"].get("as_of_date")),
        )
        universe = st.selectbox("Universe", options=["ALL", "VN30", "VNINDEX"], index=0)
        min_adv = st.number_input(
            "min_adv20_value", min_value=0.0, value=1_000_000_000.0, step=100_000_000.0
        )
        sector_in = st.text_input("sector_in (comma-separated)", value="")
        exchange_in = st.text_input("exchange_in (comma-separated)", value="")
        tags_any = st.text_input("tags_any (comma-separated)", value="")
        tags_all = st.text_input("tags_all (comma-separated)", value="")
        neutral_enabled = st.checkbox("neutralization.enabled", value=False)

        fw1, fw2 = st.columns(2)
        with fw1:
            w_value = st.number_input("w_value", value=0.2)
            w_quality = st.number_input("w_quality", value=0.2)
            w_momentum = st.number_input("w_momentum", value=0.2)
        with fw2:
            w_lowvol = st.number_input("w_lowvol", value=0.2)
            w_dividend = st.number_input("w_dividend", value=0.2)

        t1, t2 = st.columns(2)
        with t1:
            t_breakout = st.checkbox("breakout", value=True)
            t_trend = st.checkbox("trend", value=True)
        with t2:
            t_pullback = st.checkbox("pullback", value=False)
            t_vol_spike = st.checkbox("volume_spike", value=False)

        screen_payload = {
            "name": screen_name,
            "as_of_date": as_of_date.isoformat(),
            "universe": {"preset": universe},
            "filters": {
                "min_adv20_value": float(min_adv),
                "sector_in": [x.strip() for x in sector_in.split(",") if x.strip()],
                "exchange_in": [x.strip() for x in exchange_in.split(",") if x.strip()],
                "tags_any": [x.strip().lower() for x in tags_any.split(",") if x.strip()],
                "tags_all": [x.strip().lower() for x in tags_all.split(",") if x.strip()],
                "neutralization": {"enabled": neutral_enabled},
            },
            "factor_weights": {
                "value": float(w_value),
                "quality": float(w_quality),
                "momentum": float(w_momentum),
                "lowvol": float(w_lowvol),
                "dividend": float(w_dividend),
            },
            "technical_setups": {
                "breakout": t_breakout,
                "trend": t_trend,
                "pullback": t_pullback,
                "volume_spike": t_vol_spike,
            },
        }

        if st.button("Validate"):
            val = cached_post_json(
                "/screeners/validate", payload={"screen": screen_payload}, ttl_s=1
            )
            st.session_state["screen_validation"] = val
            if not val.get("errors"):
                st.session_state["screen_builder"] = val["normalized"]

        if st.button("Save screen"):
            workspaces = cached_get_json("/workspaces", params={"user_id": ""}, ttl_s=300)
            workspace_id = workspaces[0]["id"] if workspaces else None
            if workspace_id is None:
                st.error("No workspace found. Create a workspace in Watchlists page first.")
            else:
                saved = cached_post_json(
                    "/screeners/save",
                    payload={
                        "workspace_id": workspace_id,
                        "name": screen_name,
                        "screen": screen_payload,
                    },
                    ttl_s=1,
                )
                st.success(f"Saved: {saved['id']}")

    with right:
        tabs = st.tabs(["Validation", "Results", "Diff", "Explain Math"])
        with tabs[0]:
            st.markdown("#### Validation errors + normalized JSON")
            val = st.session_state.get("screen_validation")
            if val:
                if val.get("errors"):
                    st.error("; ".join(val.get("errors", [])))
                st.json(val)
            else:
                st.info("Click Validate to see normalized schema.")

        with tabs[1]:
            selected_saved = st.selectbox(
                "Saved screen (optional)",
                options=[""] + [f"{x['name']}::{x['id']}" for x in saved_screens],
            )
            saved_screen_id = selected_saved.split("::")[-1] if "::" in selected_saved else None
            if st.button("Run screen"):
                run = cached_post_json(
                    "/screeners/run",
                    payload={"screen": screen_payload, "saved_screen_id": saved_screen_id},
                    ttl_s=1,
                )
                st.session_state["last_run"] = run

            run = st.session_state.get("last_run")
            if run:
                st.caption(f"run_id={run['run_id']} reused={run['reused']}")
                results = run.get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    cols = [
                        c
                        for c in [
                            "rank",
                            "symbol",
                            "score",
                            "adv20_value",
                            "trend",
                            "breakout",
                            "pullback",
                            "volume_spike",
                            "tags",
                        ]
                        if c in df.columns
                    ]
                    st.dataframe(df[cols], use_container_width=True)
                else:
                    st.info("No results.")

        with tabs[2]:
            run = st.session_state.get("last_run")
            if run:
                st.json(run.get("diff", {}))
            else:
                st.info("Run a screen to view diff.")

        with tabs[3]:
            run = st.session_state.get("last_run")
            if run and run.get("results"):
                symbols = [r["symbol"] for r in run["results"]]
                selected = st.selectbox("Symbol", options=symbols)
                row = next((r for r in run["results"] if r["symbol"] == selected), None)
                if row:
                    st.json(row.get("explain", {}))
            else:
                st.info("Run a screen to inspect explain math.")

    st.markdown("#### Saved screens")
    if saved_screens:
        st.dataframe(
            pd.DataFrame(saved_screens)[["id", "name", "workspace_id", "updated_at"]],
            use_container_width=True,
        )
    else:
        st.info("No saved screens yet.")

    st.markdown("#### Builder JSON preview")
    st.code(json.dumps(screen_payload, ensure_ascii=False, indent=2), language="json")
