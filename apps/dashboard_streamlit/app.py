from __future__ import annotations

from apps.dashboard_streamlit.pages import _page_registry
from apps.dashboard_streamlit.ui.layout import app_shell


def _render_home() -> None:
    module = _page_registry()[0]
    module.render()


if __name__ == "__main__":
    first = _page_registry()[0]
    app_shell(first.PAGE_ID, first.PAGE_TITLE, first.render)
