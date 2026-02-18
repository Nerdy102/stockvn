from __future__ import annotations

from pathlib import Path


def test_chart_page_live_refresh_default_off() -> None:
    content = Path("apps/dashboard_streamlit/pages/2_Charting.py").read_text(encoding="utf-8")
    assert 'st.checkbox("Live refresh", value=False' in content
