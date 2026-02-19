from __future__ import annotations

from typing import Any

import streamlit as st

FONT_STACK_VI = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
            font-family: {FONT_STACK_VI};
        }}
        [data-testid="stMetric"] {{
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 12px;
            padding: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_market_today(market_today_text: list[str], as_of_date: str) -> None:
    st.subheader("1) Hôm nay thị trường")
    st.caption(f"Ngày dữ liệu mới nhất: {as_of_date}")
    for line in market_today_text[:6]:
        st.write(f"- {line}")


def render_signal_table(title: str, rows: list[dict[str, Any]], key_prefix: str) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.info("Chưa có tín hiệu phù hợp trong dữ liệu demo hiện tại.")
        return
    for idx, row in enumerate(rows[:10]):
        cols = st.columns([1, 1, 1, 3, 1.2])
        cols[0].write(str(row.get("symbol", "-")))
        cols[1].write(str(row.get("signal", "TRUNG TÍNH")))
        cols[2].write(str(row.get("confidence", "Thấp")))
        cols[3].write(str(row.get("reason", "Đang tổng hợp dữ liệu.")))
        if cols[4].button("Tạo lệnh nháp", key=f"{key_prefix}-{idx}", use_container_width=True):
            st.session_state["kiosk_prefill_symbol"] = row.get("symbol", "FPT")
            st.session_state["kiosk_prefill_model"] = row.get("model_id", "model_1")
            st.session_state["show_draft_panel"] = True
        bullets = list(row.get("reason_bullets", []))[:3]
        risk_tags = list(row.get("risk_tags", []))[:2]
        if bullets or risk_tags:
            with st.expander(f"Giải thích thêm cho {row.get('symbol', '-')}", expanded=False):
                for b in bullets:
                    st.write(f"- {b}")
                if risk_tags:
                    st.caption(f"Nhãn rủi ro: {', '.join(str(x) for x in risk_tags)}")


def render_model_cards(model_cards: list[dict[str, Any]]) -> None:
    st.subheader("3) Mô hình chạy có ổn không?")
    cols = st.columns(3)
    for idx, card in enumerate(model_cards[:3]):
        with cols[idx]:
            st.markdown(f"**{card.get('name', f'Mô hình {idx + 1}')}**")
            st.metric(
                "Kết quả 1 năm qua (giả lập)",
                f"{float(card.get('net_return_after_fees_taxes', 0.0)):.2f}%",
                help="Lợi nhuận ròng sau phí/thuế (Net return after fees/taxes)",
            )
            st.metric(
                "Sụt giảm tệ nhất",
                f"{float(card.get('max_drawdown', 0.0)):.2f}%",
                help="Mức giảm lớn nhất từ đỉnh (Max drawdown)",
            )
    st.warning("Quá khứ không đảm bảo tương lai.")


def render_paper_summary(paper: dict[str, Any]) -> None:
    st.subheader("4) Tài khoản giấy (Paper)")
    cols = st.columns(3)
    cols[0].metric("Lãi/Lỗ hiện tại", f"{float(paper.get('pnl', 0.0)):,.0f}")
    cols[1].metric("Số lệnh", str(int(paper.get("trades_count", 0))))
    cols[2].metric("Tiền mặt còn lại (%)", f"{float(paper.get('cash_pct', 0.0)):.2f}%")
