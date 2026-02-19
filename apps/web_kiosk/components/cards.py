from __future__ import annotations

from typing import Any

import streamlit as st


def render_card_hom_nay(as_of_date: str, market_brief_text_vi: list[str]) -> None:
    st.markdown('<div class="kiosk-card">', unsafe_allow_html=True)
    st.subheader("Hôm nay (Today)")
    st.caption(f"Ngày dữ liệu mới nhất: {as_of_date}")
    for line in market_brief_text_vi[:6]:
        st.write(f"• {line}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_card_tin_hieu(rows_buy: list[dict[str, Any]], rows_sell: list[dict[str, Any]]) -> None:
    st.markdown('<div class="kiosk-card">', unsafe_allow_html=True)
    st.subheader("Tín hiệu rõ ràng (Clear signals)")
    tab_buy, tab_sell = st.tabs(["Có thể MUA (nháp)", "Có thể BÁN (nháp)"])

    def _render_rows(rows: list[dict[str, Any]], side: str, key_prefix: str) -> None:
        if not rows:
            st.info("Chưa có tín hiệu phù hợp.")
            return
        for i, row in enumerate(rows[:10]):
            cols = st.columns([1, 1, 1, 3, 1.2])
            cols[0].write(str(row.get("symbol", "-")))
            cols[1].write(str(row.get("signal", side)))
            cols[2].write(str(row.get("confidence", "Thấp")))
            cols[3].write(str(row.get("reason_short", row.get("reason", "Tín hiệu tổng hợp."))))
            if cols[4].button("Tạo lệnh nháp", key=f"{key_prefix}-{i}", use_container_width=True):
                st.session_state["selected_signal"] = {
                    "symbol": row.get("symbol", "FPT"),
                    "side": side,
                    "model_id": row.get("model_id", "model_1"),
                    "reason_short": row.get("reason_short", row.get("reason", "")),
                    "confidence": row.get("confidence", "Thấp"),
                    "risk_tags": row.get("risk_tags", []),
                }
                st.success(f"Đã chọn tín hiệu {row.get('symbol')} để tạo nháp.")

    with tab_buy:
        _render_rows(rows_buy, "BUY", "buy")
    with tab_sell:
        _render_rows(rows_sell, "SELL", "sell")
    st.markdown('</div>', unsafe_allow_html=True)


def render_card_readiness(readiness: dict[str, Any], advanced: dict[str, Any], health: dict[str, Any]) -> None:
    st.markdown('<div class="kiosk-card">', unsafe_allow_html=True)
    st.subheader("Độ sẵn sàng (Readiness)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ổn định (Stability)", f"{float(readiness.get('stability_score', 0.0)):.1f}/100")
    c2.metric(
        "Kịch bản xấu nhất (Worst case)",
        f"Lãi ròng (Net) {float(readiness.get('worst_case_net_return', 0.0)):.2f} | Sụt giảm tối đa (MDD) {float(readiness.get('worst_case_mdd', 0.0)):.2f}",
    )
    state = "Đang bật công tắc khẩn cấp (Kill-switch)" if readiness.get("kill_switch_state") else ("Bình thường" if str(readiness.get("drift_state", "OK")).upper()=="OK" else str(readiness.get("drift_state", "Tạm dừng")))
    c3.metric("Trạng thái hệ thống", state)

    st.caption(
        f"Cơ sở dữ liệu (DB): {'Tốt' if health.get('db_ok') else 'Lỗi'} • Độ mới dữ liệu: {'Tốt' if health.get('data_freshness_ok') else 'Cũ'} • Đối soát: {health.get('last_reconcile_ts','Không có')}"
    )
    st.write(f"Tin cậy thống kê: {readiness.get('tin_cay_thong_ke', 'Thấp')}")
    st.write(f"Rủi ro chọn nhầm mô hình (PBO): {readiness.get('pbo_bucket', 'Chưa đủ mẫu')}")
    p_spa = readiness.get('rc_spa_p')
    st.write(f"Kiểm định soi dữ liệu (data snooping, RC/SPA): p={('Không có' if p_spa is None else f'{float(p_spa):.3f}')} (tham khảo)")

    with st.expander("Xem thêm"):
        st.write(f"Mã báo cáo (Report ID): {readiness.get('report_id','Không có')}")
        st.json({"hashes": readiness.get("hashes", {})})
        st.write("Tóm tắt walk-forward folds")
        st.dataframe(advanced.get("walk_forward_fold_summary", []), use_container_width=True)
        st.write("Bảng stress")
        st.dataframe(advanced.get("stress_table", []), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
