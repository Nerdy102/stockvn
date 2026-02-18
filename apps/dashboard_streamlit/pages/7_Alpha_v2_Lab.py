from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from core.report_pack_v3 import export_report_pack_v3

from apps.dashboard_streamlit.ui.cache import cached_post_json

PAGE_ID = "alpha_v2_lab"
PAGE_TITLE = "Alpha v2 Lab"


def render() -> None:
    if st.button("Run diagnostics v2"):
        st.json(cached_post_json("/ml/diagnostics", payload={}, ttl_s=60))
    if st.button("Run backtest v2"):
        b = cached_post_json("/ml/backtest", payload={"mode": "v2"}, ttl_s=60)
        wf = b.get("walk_forward", {})
        curve = pd.DataFrame(wf.get("equity_curve", []))
        if not curve.empty and "equity" in curve:
            st.line_chart(
                curve.set_index("as_of_date")["equity"]
                if "as_of_date" in curve
                else curve["equity"]
            )

    st.markdown("---")
    st.subheader("Audit Export — Report Pack v3")
    export_run_id = st.text_input("run_id", value="")
    if st.button("Export report pack v3"):
        if not export_run_id.strip():
            st.warning("Please provide a run_id before exporting.")
        else:
            bundle = export_report_pack_v3(export_run_id.strip())
            st.success(f"Export completed: {bundle.outdir}")
            for name, p in {
                "Download HTML": bundle.html_path,
                "Download PDF": bundle.pdf_path,
                "Download Manifest": bundle.manifest_path,
            }.items():
                p = Path(p)
                st.download_button(name, data=p.read_bytes(), file_name=p.name)
