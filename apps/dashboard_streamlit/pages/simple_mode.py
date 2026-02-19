from __future__ import annotations

import streamlit as st

from apps.dashboard_streamlit.lib import api

MAX_POINTS_PER_CHART = 300


def render() -> None:
    st.title("🚀 Giao dịch đơn giản")
    st.caption("Công cụ giáo dục: không phải lời khuyên đầu tư, có thể thua lỗ.")

    tab_main, tab_compare = st.tabs(["Wizard 3 bước", "📊 So sánh Model 1/2/3"])

    with tab_main:
        st.subheader("Bước 1 — Chọn mã & chế độ")
        symbol = st.text_input("Mã cổ phiếu", value="FPT").upper().strip()
        timeframe = st.selectbox("Timeframe", ["1D", "60m"], index=0)
        mode = st.selectbox("Chế độ chạy", ["paper", "draft"])
        if st.button("Đồng bộ dữ liệu"):
            status = api.get("/simple/sync_status", {"symbol": symbol, "timeframe": timeframe})
            st.json(status)

        st.subheader("Bước 2 — Chọn model & chạy")
        model = st.radio(
            "Model",
            ["model_1", "model_2", "model_3"],
            format_func=lambda x: {
                "model_1": "Model 1 — Xu hướng",
                "model_2": "Model 2 — Hồi quy về trung bình",
                "model_3": "Model 3 — Kết hợp Factor + Regime",
            }[x],
        )

        if st.button("Chạy phân tích"):
            resp = api.post(
                "/simple/run_signal",
                {"symbol": symbol, "timeframe": timeframe, "model_id": model, "mode": mode},
            )
            st.session_state["simple_last"] = resp

        last = st.session_state.get("simple_last")
        if last:
            signal = last["signal"]
            draft = last.get("draft")
            st.success(f"Tín hiệu: {signal['signal']} | Độ tin cậy: {signal['confidence']}")
            st.write("Giải thích ngắn:")
            for line in signal["explanation"]:
                st.write(f"- {line}")
            st.write("Rủi ro:")
            for line in signal["risks"]:
                st.write(f"- {line}")
            if draft:
                st.subheader("Bước 3 — Gợi ý lệnh & xác nhận")
                st.json(draft)
                ack1 = st.checkbox("Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư")
                ack2 = st.checkbox("Tôi hiểu có thể thua lỗ")
                if st.button("XÁC NHẬN THỰC HIỆN"):
                    out = api.post(
                        "/simple/confirm_execute",
                        {
                            "portfolio_id": 1,
                            "mode": mode,
                            "acknowledged_educational": ack1,
                            "acknowledged_loss": ack2,
                            "draft": draft,
                        },
                    )
                    st.json(out)

    with tab_compare:
        symbols = st.text_input("Danh sách mã (phân tách dấu phẩy)", value="FPT,VNM,VCB")
        lookback = st.slider("Số phiên backtest", 60, 756, 252)
        if st.button("Chạy so sánh"):
            rows = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            resp = api.post(
                "/simple/run_compare",
                {"symbols": rows, "lookback_days": lookback, "timeframe": "1D"},
            )
            st.warning(resp["warning"])
            st.dataframe(resp["leaderboard"], use_container_width=True)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
