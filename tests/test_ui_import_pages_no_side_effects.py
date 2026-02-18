from __future__ import annotations

from pathlib import Path

import streamlit as st

from apps.dashboard_streamlit.pages import _page_registry


def test_import_pages_no_set_page_config_side_effects(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("st.set_page_config should not be called when importing pages")

    monkeypatch.setattr(st, "set_page_config", _boom)

    pages = _page_registry()
    assert pages
    for page in pages:
        assert hasattr(page, "PAGE_ID")
        assert hasattr(page, "PAGE_TITLE")
        assert hasattr(page, "render")
