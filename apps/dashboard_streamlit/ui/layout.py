from __future__ import annotations

import datetime as dt
import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path

import streamlit as st

from apps.dashboard_streamlit.ui.cache import cached_get_json
from apps.dashboard_streamlit.ui.realtime_poll import poll_summary
from apps.dashboard_streamlit.ui.text import DISCLAIMER_SHORT
from apps.dashboard_streamlit.ui.theme import apply_theme

_PAGE_CONFIG_SET = "_ui_page_config_set"

NAV_ORDER = [
    "🏠 Tổng quan hôm nay",
    "Watchlists",
    "Screener",
    "Chart",
    "Portfolio",
    "Alerts",
    "Research Lab",
    "Data Health",
    "Settings",
    "New Orders",
]

PAGE_PATHS = {
    "🏠 Tổng quan hôm nay": "apps/dashboard_streamlit/pages/0_Home_Dashboard.py",
    "Watchlists": "apps/dashboard_streamlit/pages/3_Heatmap.py",
    "Screener": "apps/dashboard_streamlit/pages/1_Screener.py",
    "Chart": "apps/dashboard_streamlit/pages/2_Charting.py",
    "Portfolio": "apps/dashboard_streamlit/pages/4_Portfolio.py",
    "Alerts": "apps/dashboard_streamlit/pages/5_Alerts.py",
    "Research Lab": "apps/dashboard_streamlit/pages/6_ML_Lab.py",
    "Data Health": "apps/dashboard_streamlit/pages/8_Data_Health.py",
    "Settings": "apps/dashboard_streamlit/pages/9_Settings.py",
    "New Orders": "apps/dashboard_streamlit/pages/10_New_Orders.py",
}


def _set_page_config_once() -> None:
    if st.session_state.get(_PAGE_CONFIG_SET):
        return
    st.set_page_config(page_title="VN Invest Toolkit", layout="wide")
    st.session_state[_PAGE_CONFIG_SET] = True


def _load_latest_date() -> dt.date | None:
    try:
        payload = cached_get_json("/prices/latest_date", params=None, ttl_s=300)
        raw = payload.get("latest_date") if isinstance(payload, dict) else payload
        if raw:
            return dt.date.fromisoformat(str(raw))
    except Exception:
        logging.exception("latest_date API failed; fallback will be used")
    return None


def _load_data_health_summary() -> str:
    try:
        payload = cached_get_json("/data/health/summary", params=None, ttl_s=60)
        if isinstance(payload, dict):
            freshness = payload.get("freshness")
            status = payload.get("status")
            if freshness or status:
                return f"status={status or 'n/a'} · freshness={freshness or 'n/a'}"
    except Exception:
        logging.exception("data health summary API failed; fallback to /health")
    try:
        payload = cached_get_json("/health", params=None, ttl_s=60)
        if isinstance(payload, dict):
            return f"health={payload.get('status', 'ok')}"
    except Exception:
        logging.exception("health API failed")
    return "health=unknown"


def _fallback_demo_last_date() -> dt.date:
    return dt.date(2025, 12, 31)


def topbar(page_id: str) -> None:
    with st.container():
        st.markdown('<div class="topbar-wrap">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 2, 4])
        with c1:
            mode = st.selectbox("Mode", options=["DEMO", "PAPER", "LIVE"], index=0, key="ui_mode")
            st.caption(f"Mode: {mode}")
        with c2:
            default_date = _load_latest_date()
            if default_date is None:
                default_date = _fallback_demo_last_date()
                logging.warning(
                    "latest_date missing; using demo fallback", extra={"page_id": page_id}
                )
            as_of = st.date_input("As of", value=default_date, key="as_of_date")
            st.caption(f"As of: {as_of.isoformat()}")
        with c3:
            st.caption(f"Data freshness: {_load_data_health_summary()}")
            realtime_enabled = bool(st.session_state.get("realtime_enabled", False))
            summary = poll_summary(ui_mode=mode, realtime_enabled=realtime_enabled)
            if summary.get("enabled"):
                lag = summary.get("stream_lag_s")
                throttled = bool(summary.get("throttled", False))
                badge = "🟢 realtime"
                if throttled:
                    badge = "🟠 realtime throttled"
                if lag is not None:
                    st.caption(f"{badge} · lag={lag}s · poll={summary.get('interval_s', 2)}s")
                else:
                    st.caption(f"{badge} · poll={summary.get('interval_s', 2)}s")
            elif summary.get("realtime_disabled"):
                st.caption("⚪ realtime off")
        with c4:
            st.markdown(
                f'<div class="disclaimer">⚠️ {DISCLAIMER_SHORT}</div>', unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)


def sidebar_nav() -> str:
    current = st.session_state.get("page", NAV_ORDER[0])
    idx = NAV_ORDER.index(current) if current in NAV_ORDER else 0
    selected = st.sidebar.radio("Navigation", options=NAV_ORDER, index=idx)
    st.session_state["page"] = selected
    return selected


def load_page_module(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load page module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def app_shell(page_id: str, title: str, render_fn: Callable[[], None]) -> None:
    _set_page_config_once()
    apply_theme()
    topbar(page_id)
    selected = sidebar_nav()
    st.title(title)
    try:
        if selected in PAGE_PATHS and selected != title:
            module = load_page_module(PAGE_PATHS[selected])
            module.render()
            return
        render_fn()
    except Exception:
        logging.exception("page render failed", extra={"page_id": page_id})
        st.error(
            "Đã xảy ra lỗi khi tải trang. Vui lòng thử lại với khoảng dữ liệu nhỏ hơn hoặc mở logs để chẩn đoán."
        )
        st.info("Gợi ý: kiểm tra terminal chạy Streamlit để xem stacktrace chi tiết.")
