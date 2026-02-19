from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from apps.dashboard_streamlit.lib import api
from apps.web_kiosk.components import (
    inject_theme,
    render_action_bar,
    render_card_hom_nay,
    render_card_readiness,
    render_card_tin_hieu,
)
from apps.web_kiosk.demo_data import fallback_payload


def _advanced_ui_enabled() -> bool:
    return os.getenv("ENABLE_ADVANCED_UI", "false").strip().lower() == "true"


def _load_kiosk_payload() -> dict[str, Any]:
    try:
        return api.get(
            "/simple/kiosk_v3",
            params={"universe": "VN30", "limit_signals": 10, "lookback": 252},
        )
    except (httpx.HTTPError, ValueError):
        st.warning("Không kết nối được API, đang dùng dữ liệu demo offline.")
        old = fallback_payload()
        return {
            "as_of_date": str(old.get("as_of_date", "-")),
            "market_brief_text_vi": list(old.get("market_today_text", [])),
            "buy_candidates": list(old.get("buy_candidates", []))[:10],
            "sell_candidates": list(old.get("sell_candidates", []))[:10],
            "readiness_summary": {
                "stability_score": 0.0,
                "worst_case_net_return": 0.0,
                "worst_case_mdd": 0.0,
                "drift_state": "OK",
                "kill_switch_state": False,
                "paused_reason_code": None,
                "report_id": "offline-demo",
                "hashes": {},
            },
            "system_health_summary": {
                "db_ok": True,
                "data_freshness_ok": True,
                "last_reconcile_ts": None,
            },
            "advanced": {"stress_table": [], "walk_forward_fold_summary": []},
        }


def render() -> None:
    st.set_page_config(page_title="Kiosk UI v3", page_icon="🧭", layout="wide")
    inject_theme()

    st.title("Kiosk UI v3: siêu tối giản")
    st.caption("Một màn hình duy nhất: nhìn 1 phát hiểu ngay, không giao dịch tự động.")
    st.info("Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư.")

    if _advanced_ui_enabled():
        st.link_button(
            "Mở giao diện nâng cao (Advanced UI)",
            os.getenv("ADVANCED_UI_URL", "http://localhost:8501"),
        )

    payload = _load_kiosk_payload()

    render_card_hom_nay(
        as_of_date=str(payload.get("as_of_date", "-")),
        market_brief_text_vi=list(payload.get("market_brief_text_vi", [])),
    )
    render_card_tin_hieu(
        rows_buy=list(payload.get("buy_candidates", [])),
        rows_sell=list(payload.get("sell_candidates", [])),
    )
    render_card_readiness(
        readiness=dict(payload.get("readiness_summary", {})),
        advanced=dict(payload.get("advanced", {})),
        health=dict(payload.get("system_health_summary", {})),
    )
    render_action_bar(as_of_date=str(payload.get("as_of_date", "-")))


if __name__ == "__main__":
    render()
