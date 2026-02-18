from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from apps.dashboard_streamlit.lib.api import api_base, get_bytes
from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json

PAGE_ID = "watchlists"
PAGE_TITLE = "Watchlists"


def _tickers_universe() -> list[dict]:
    return cached_get_json("/tickers", params={"limit": 2000, "offset": 0}, ttl_s=1800)


def render() -> None:
    st.subheader("Workspaces & Watchlists v2")

    workspaces = cached_get_json("/workspaces", params={"user_id": ""}, ttl_s=300)
    ws_col1, ws_col2 = st.columns([2, 1])
    with ws_col1:
        new_ws_name = st.text_input("New workspace name", value="Default")
    with ws_col2:
        if st.button("Create workspace"):
            cached_post_json(
                "/workspaces", payload={"user_id": None, "name": new_ws_name}, ttl_s=60
            )
            st.rerun()

    if not workspaces:
        st.info("No workspace yet. Create one to continue.")
        return

    ws_map = {f"{w['name']} ({w['id'][:8]})": w for w in workspaces}
    selected_ws_label = st.selectbox("Workspace", options=list(ws_map.keys()))
    selected_ws = ws_map[selected_ws_label]

    watchlists = cached_get_json(
        f"/workspaces/{selected_ws['id']}/watchlists", params=None, ttl_s=300
    )
    wl_col1, wl_col2 = st.columns([2, 1])
    with wl_col1:
        new_wl_name = st.text_input("New watchlist name", value="Main Watchlist")
    with wl_col2:
        if st.button("Create watchlist"):
            cached_post_json(
                f"/workspaces/{selected_ws['id']}/watchlists",
                payload={"name": new_wl_name},
                ttl_s=60,
            )
            st.rerun()

    if not watchlists:
        st.info("No watchlist yet. Create one to continue.")
        return

    wl_map = {f"{w['name']} ({w['id'][:8]})": w for w in watchlists}
    selected_wl_label = st.selectbox("Watchlist", options=list(wl_map.keys()))
    selected_wl = wl_map[selected_wl_label]

    tickers = _tickers_universe()
    symbols = [t["symbol"] for t in tickers]
    tag_dictionary = cached_get_json("/tag_dictionary", params=None, ttl_s=1800)
    taxonomy = [t["tag"] for t in tag_dictionary]

    st.markdown("#### Add symbol")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        symbol = st.selectbox("Symbol", options=symbols, index=0)
    with c2:
        tags = st.multiselect("Tags", options=taxonomy, default=[])
    with c3:
        pinned = st.checkbox("Pinned", value=False)
    note = st.text_area("Note", value="", max_chars=2000)
    if st.button("Upsert item"):
        res = cached_post_json(
            f"/watchlists/{selected_wl['id']}/items",
            payload={"symbol": symbol, "tags": tags, "note": note, "pinned": pinned},
            ttl_s=60,
        )
        st.success(res["status"])
        st.rerun()

    st.markdown("#### Bulk paste symbols")
    bulk_input = st.text_area("Paste symbols (comma/newline separated)", value="")
    if st.button("Bulk upsert") and bulk_input.strip():
        chunks = [x.strip().upper() for x in bulk_input.replace("\n", ",").split(",") if x.strip()]
        for sym in chunks:
            cached_post_json(
                f"/watchlists/{selected_wl['id']}/items",
                payload={"symbol": sym, "tags": [], "note": "", "pinned": False},
                ttl_s=60,
            )
        st.success(f"Processed {len(chunks)} symbols")
        st.rerun()

    st.markdown("#### Import CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None and st.button("Import CSV"):
        import httpx

        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        base = api_base()
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"{base.rstrip('/')}/watchlists/{selected_wl['id']}/import", files=files
            )
            r.raise_for_status()
            report = r.json()
        st.json(report)

    st.markdown("#### Items")
    items = cached_get_json(f"/watchlists/{selected_wl['id']}/items", params=None, ttl_s=60)
    if items:
        df = pd.DataFrame(items)
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(lambda v: ",".join(v) if isinstance(v, list) else str(v))

        display_cols = ["id", "symbol", "exchange", "tags", "note", "pinned"] if "exchange" in df.columns else ["id", "symbol", "tags", "note", "pinned"]
        edited = st.data_editor(
            df[display_cols],
            use_container_width=True,
            num_rows="fixed",
            disabled=["id", "symbol"],
        )
        if st.button("Save table edits"):
            for row in edited.to_dict(orient="records"):
                tags_list = [t.strip() for t in str(row["tags"]).split(",") if t.strip()]
                import httpx

                base = api_base()
                with httpx.Client(timeout=30) as client:
                    r = client.patch(
                        f"{base.rstrip('/')}/watchlists/{selected_wl['id']}/items/{row['id']}",
                        json={
                            "tags": tags_list,
                            "note": row["note"],
                            "pinned": bool(row["pinned"]),
                        },
                    )
                    r.raise_for_status()
            st.success("Updated items")
            st.rerun()

    st.markdown("#### Export CSV")
    csv_bytes = get_bytes(f"/watchlists/{selected_wl['id']}/export")
    st.download_button(
        "Download export",
        data=io.BytesIO(csv_bytes),
        file_name=f"watchlist_{selected_wl['id'][:8]}.csv",
        mime="text/csv",
    )
