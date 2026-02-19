from __future__ import annotations


def test_ui_simple_mode_import() -> None:
    import apps.dashboard_streamlit.pages.simple_mode as page

    assert hasattr(page, "render")
