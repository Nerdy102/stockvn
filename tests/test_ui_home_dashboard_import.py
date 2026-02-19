from __future__ import annotations

import importlib.util
from pathlib import Path


def test_ui_home_dashboard_import() -> None:
    path = Path("apps/dashboard_streamlit/pages/0_Home_Dashboard.py")
    spec = importlib.util.spec_from_file_location("home_dashboard_page", path)
    assert spec is not None and spec.loader is not None
