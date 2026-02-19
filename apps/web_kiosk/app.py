from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from apps.dashboard_streamlit.lib import api
from apps.web_kiosk.components import (
    inject_theme,
    render_market_today,
    render_model_cards,
    render_paper_summary,
    render_signal_table,
)
from apps.web_kiosk.demo_data import fallback_payload


def _advanced_ui_enabled() -> bool:
    return os.getenv("ENABLE_ADVANCED_UI", "false").strip().lower() == "true"


def _load_kiosk_payload() -> dict[str, Any]:
    try:
        return api.get("/simple/kiosk", params={"universe": "VN30", "limit_signals": 10})
    except (httpx.HTTPError, ValueError):
        st.warning("Không kết nối được API, đang dùng dữ liệu demo offline.")
        return fallback_payload()


def _render_draft_panel() -> None:
    with st.expander(
        "Tạo và xác nhận lệnh nháp", expanded=bool(st.session_state.get("show_draft_panel", False))
    ):
        symbol = st.text_input(
            "Mã",
            value=str(st.session_state.get("kiosk_prefill_symbol", "FPT")).upper(),
            key="kiosk_symbol",
        )
        model_id = st.selectbox(
            "Mô hình",
            ["model_1", "model_2", "model_3"],
            index=["model_1", "model_2", "model_3"].index(
                str(st.session_state.get("kiosk_prefill_model", "model_1"))
                if str(st.session_state.get("kiosk_prefill_model", "model_1"))
                in {"model_1", "model_2", "model_3"}
                else "model_1"
            ),
        )
        age = int(st.number_input("Tuổi", min_value=10, max_value=100, value=18, step=1))

        draft_data = st.session_state.get("kiosk_draft_data")
        if draft_data and draft_data.get("symbol") == symbol:
            draft = draft_data
            st.markdown("**Bạn sắp làm gì**")
            st.write(
                f"- Bạn sẽ {'MUA' if draft.get('side') == 'BUY' else 'BÁN'} (nháp) mã {symbol}"
            )
            st.write(f"- Khối lượng: {draft.get('qty')} • Giá dự kiến: {draft.get('price')}")
            st.write(
                f"- Ước tính phí/thuế/trượt giá: {draft.get('fee_tax', {}).get('total_cost', 0)}"
            )

        ack_loss = st.checkbox("Tôi hiểu có thể thua lỗ (Risk of loss)", key="kiosk_ack_loss")
        ack_edu = st.checkbox(
            "Tôi hiểu đây không phải lời khuyên đầu tư (Not investment advice)",
            key="kiosk_ack_edu",
        )
        ack_live = st.checkbox(
            "Tôi đủ điều kiện pháp lý và không yêu cầu hướng dẫn lách luật",
            key="kiosk_ack_legal",
        )

        if st.button("Xác nhận thực hiện (Confirm execute)", use_container_width=True):
            try:
                resp = api.post(
                    "/simple/run_signal",
                    {
                        "symbol": symbol,
                        "timeframe": "1D",
                        "model_id": model_id,
                        "mode": "draft",
                        "market": "vn",
                        "trading_type": "spot_paper",
                    },
                )
                draft = resp.get("draft")
                if not draft:
                    st.error("Không tạo được lệnh nháp từ tín hiệu hiện tại.")
                    return
                st.session_state["kiosk_draft_data"] = draft
                out = api.post(
                    "/simple/confirm_execute",
                    {
                        "portfolio_id": 1,
                        "user_id": "kiosk-user",
                        "session_id": "kiosk-session",
                        "idempotency_token": f"kiosk-{symbol}-{model_id}",
                        "mode": "draft",
                        "acknowledged_educational": ack_edu,
                        "acknowledged_loss": ack_loss,
                        "acknowledged_live_eligibility": ack_live,
                        "age": age,
                        "draft": draft,
                    },
                )
                st.success("Đã xác nhận và lưu lệnh nháp thành công.")
                st.json(out)
            except Exception as exc:
                st.error(f"Không thể xác nhận lệnh. Mã lỗi/chi tiết: {exc}")


def render() -> None:
    st.set_page_config(page_title="Kiosk đơn giản", page_icon="🏠", layout="wide")
    inject_theme()

    st.title("🏠 Hôm nay")
    st.caption("Giao diện siêu đơn giản: mở web là thấy ngay tín hiệu và thao tác chính.")
    st.info("Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư.")

    if _advanced_ui_enabled():
        st.link_button(
            "Mở giao diện nâng cao (Advanced)",
            os.getenv("ADVANCED_UI_URL", "http://localhost:8501"),
        )

    payload = _load_kiosk_payload()

    cta1, cta2 = st.columns(2)
    if cta1.button("Xem tín hiệu hôm nay", type="primary", use_container_width=True):
        st.session_state["show_signal_panel"] = True
    if cta2.button("Tạo lệnh nháp", type="primary", use_container_width=True):
        st.session_state["show_draft_panel"] = True

    col_a, col_b = st.columns(2)
    with col_a:
        render_market_today(
            payload.get("market_today_text", []), str(payload.get("as_of_date", "-"))
        )
    with col_b:
        st.subheader("2) Tín hiệu rõ ràng")
        if st.session_state.get("show_signal_panel", False):
            render_signal_table("Có thể MUA (nháp)", payload.get("buy_candidates", []), "buy")
            render_signal_table("Có thể BÁN (nháp)", payload.get("sell_candidates", []), "sell")
        else:
            st.caption("Bấm nút “Xem tín hiệu hôm nay” để mở danh sách gợi ý.")

    render_model_cards(payload.get("model_cards", []))
    render_paper_summary(payload.get("paper_summary", {}))
    _render_draft_panel()

    with st.expander("Xem thêm chi tiết nâng cao (Advanced details)"):
        st.json(payload)


if __name__ == "__main__":
    render()
