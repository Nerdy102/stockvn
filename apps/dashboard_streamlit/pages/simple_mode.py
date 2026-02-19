from __future__ import annotations

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from apps.dashboard_streamlit.lib import api

MAX_POINTS_PER_CHART = 300
FONT_STACK_VI = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'


def _render_chart(chart_points: list[dict[str, float | str]], marker_time: str | None) -> None:
    if not chart_points:
        st.info("Chưa có dữ liệu biểu đồ tối giản (Minimal chart).")
        return
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3]
    )
    x = [row["time"] for row in chart_points]
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=[row["open"] for row in chart_points],
            high=[row["high"] for row in chart_points],
            low=[row["low"] for row in chart_points],
            close=[row["close"] for row in chart_points],
            name="Nến (Candlestick)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=[row["ema20"] for row in chart_points], name="EMA20", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=[row["ema50"] for row in chart_points], name="EMA50", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=[row["volume"] for row in chart_points], name="Khối lượng (Volume)"),
        row=2,
        col=1,
    )
    if marker_time:
        marker_row = next(
            (row for row in chart_points if str(row["time"]) == str(marker_time)),
            chart_points[-1],
        )
        fig.add_trace(
            go.Scatter(
                x=[marker_row["time"]],
                y=[marker_row["close"]],
                mode="markers",
                marker={"size": 12, "symbol": "diamond"},
                name="Điểm tín hiệu gần nhất (Signal marker)",
            ),
            row=1,
            col=1,
        )
    fig.update_layout(height=620, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
            font-family: {FONT_STACK_VI};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🚀 Giao dịch đơn giản (Simple Trading)")
    st.caption(
        "Không phải lời khuyên đầu tư (Not investment advice) • Quá khứ không đảm bảo tương lai (Past performance is not indicative of future results) • Có thể thua lỗ (Risk of loss)."
    )
    st.info(
        "Kiểm tra hiển thị dấu: Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư."
    )
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = (
            f"streamlit-{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
        )
    if "session_user_id" not in st.session_state:
        st.session_state["session_user_id"] = "streamlit-user"
    if "idempotency_token" not in st.session_state:
        st.session_state["idempotency_token"] = ""

    meta: dict[str, object] = {"live_enabled": False, "max_points_per_chart": MAX_POINTS_PER_CHART}
    api_ready = True
    try:
        meta = api.get("/simple/models")
    except (httpx.HTTPError, ValueError):
        api_ready = False
        st.warning(
            "Chưa kết nối được API Simple Mode. Bạn vẫn có thể xem giao diện; hãy chạy API để thực hiện phân tích."
        )

    tab_main, tab_compare = st.tabs(
        ["Luồng 3 bước (3-step wizard)", "📊 So sánh Mô hình 1/2/3 (Model comparison)"]
    )

    with tab_main:
        st.subheader("Bước 1 — Chọn mã & chế độ")
        market = st.selectbox(
            "Thị trường (Market)",
            ["Cổ phiếu Việt Nam (VN Stocks)", "Tiền mã hoá (Crypto)"],
            index=0,
        )
        is_crypto = market.startswith("Tiền mã hoá")
        default_symbol = str(
            st.session_state.get("simple_prefill_symbol", "BTC" if is_crypto else "FPT")
        )
        symbol = st.text_input("Mã giao dịch (Symbol)", value=default_symbol).upper().strip()
        trading_type = "spot_paper"
        if is_crypto:
            trading_type = st.selectbox(
                "Loại giao dịch (Trading type)",
                ["spot_paper", "perp_paper"],
                index=0,
                format_func=lambda x: (
                    "Giao ngay — giao dịch giấy (Spot paper)"
                    if x == "spot_paper"
                    else "Hợp đồng vĩnh cửu — giao dịch giấy (Perp paper, Long/Short)"
                ),
            )
            exchange = st.selectbox(
                "Sàn dữ liệu (Exchange)",
                ["binance_public"],
                index=0,
                format_func=lambda x: "Binance công khai (Binance public)",
            )
        else:
            exchange = st.selectbox(
                "Sàn (Exchange)", ["Tự nhận diện", "HOSE", "HNX", "UPCOM"], index=0
            )
        default_tf = str(st.session_state.get("simple_prefill_timeframe", "1D"))
        timeframe_options = ["1D", "60m"]
        timeframe = st.selectbox(
            "Khung thời gian (Timeframe)",
            timeframe_options,
            index=(timeframe_options.index(default_tf) if default_tf in timeframe_options else 0),
        )
        modes = ["paper", "draft"]
        mode_labels = {
            "paper": "Giao dịch giấy (Paper trading)",
            "draft": "Lệnh nháp (Order draft)",
            "live": "Giao dịch thật (Live trading)",
        }
        if bool(meta.get("live_enabled")):
            modes.append("live")
        mode = st.selectbox("Chế độ chạy (Mode)", modes, format_func=lambda x: mode_labels[x])
        if mode == "live":
            st.warning(
                "Bạn đang ở chế độ giao dịch thật (Live trading). Luôn kiểm tra hạn mức rủi ro trước khi xác nhận."
            )
            cks1, cks2 = st.columns(2)
            with cks1:
                if st.button("DỪNG KHẨN CẤP (Kill-switch)", disabled=not api_ready):
                    out = api.post("/simple/kill_switch/toggle", {"enabled": True})
                    st.error(f"Kill-switch: {out.get('status','PAUSED')}")
            with cks2:
                if st.button("MỞ LẠI GIAO DỊCH (Tắt kill-switch)", disabled=not api_ready):
                    out = api.post("/simple/kill_switch/toggle", {"enabled": False})
                    st.success(f"Kill-switch: {out.get('status','RUNNING')}")
        if st.button("Đồng bộ dữ liệu (Sync data)") and api_ready:
            status = api.get("/simple/sync_status", {"symbol": symbol, "timeframe": timeframe})
            st.write(
                f"Trạng thái dữ liệu: {status['rows']} thanh giá (bars) • Cập nhật gần nhất: {status['last_update'] or 'Không có (N/A)'}"
            )
            if status.get("missing"):
                st.warning(status["missing"])
        st.caption(
            f"Sàn mặc định khi không nhận diện được: {exchange if exchange != 'Tự nhận diện' else 'HOSE'}"
        )

        st.subheader("Bước 2 — Chọn mô hình & chạy")
        preferred_model = st.session_state.get("simple_preferred_model", "model_1")
        model_list = ["model_1", "model_2", "model_3"]
        default_model_index = (
            model_list.index(preferred_model) if preferred_model in model_list else 0
        )
        model = st.radio(
            "Bộ mô hình (Model Zoo)",
            model_list,
            index=default_model_index,
            format_func=lambda x: {
                "model_1": "Mô hình 1 — Xu hướng (Trend-following)",
                "model_2": "Mô hình 2 — Hồi quy về trung bình (Mean-reversion)",
                "model_3": "Mô hình 3 — Kết hợp nhân tố + chế độ thị trường (Factor + Regime)",
            }[x],
        )

        if st.button("Chạy phân tích (Run analysis)", disabled=not api_ready):
            resp = api.post(
                "/simple/run_signal",
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "model_id": model,
                    "mode": mode,
                    "market": "crypto" if is_crypto else "vn",
                    "trading_type": trading_type,
                    "exchange": exchange,
                },
            )
            st.session_state["simple_last"] = resp

        last = st.session_state.get("simple_last")
        if last:
            signal = last["signal"]
            draft = last.get("draft")
            st.success(
                f"Kết luận hiện tại (Current view) — Tín hiệu (Signal): {signal['signal']} | Độ tin cậy (Confidence): {signal['confidence']}"
            )
            st.write("Giải thích ngắn (Short explanation):")
            for line in signal["explanation"]:
                st.write(f"- {line}")
            st.write("Rủi ro (Risks):")
            for line in signal["risks"]:
                st.write(f"- {line}")
            st.write(
                f"Ngân sách biểu đồ: tối đa {MAX_POINTS_PER_CHART} điểm (MAX_POINTS_PER_CHART), API trả về {last.get('data_status', {}).get('rows', 0)} điểm."
            )
            _render_chart(last.get("chart", []), signal.get("marker_time"))
            if draft:
                st.subheader("Giả lập phí/thuế (Fee/Tax simulation)")
                st.table(
                    [
                        {
                            "Phí giao dịch (Commission)": draft["fee_tax"]["commission"],
                            "Thuế bán (Sell tax)": draft["fee_tax"]["sell_tax"],
                            "Phí trượt giá (Slippage)": draft["fee_tax"]["slippage_est"],
                            "Tổng chi phí (Total cost)": draft["fee_tax"]["total_cost"],
                        }
                    ]
                )
                st.subheader("Bước 3 — Gợi ý lệnh & xác nhận")
                st.write(
                    f"Hành động nháp: {'MUA (nháp)' if draft['side'] == 'BUY' else ('Mở vị thế bán (Short) (nháp)' if draft['side'] == 'SHORT' else 'BÁN (nháp)')}"
                )
                st.write(
                    f"Khối lượng đề xuất: {draft['qty']} cổ phiếu • Giá giả lập: {draft['price']} • Giá trị lệnh: {draft['notional']}"
                )
                st.write("Lý do (Rules triggered):")
                for reason in draft["reasons"]:
                    st.write(f"- {reason}")
                st.write("Rủi ro giao dịch (Trading risks):")
                for risk in draft["risks"]:
                    st.write(f"- {risk}")

                if mode == "live":
                    st.subheader("Bước 3.5 — Xác nhận giao dịch thật (Live confirmation)")
                    st.warning(
                        "Bạn sắp xác nhận lệnh thật. Hãy kiểm tra kỹ thông tin lệnh, chi phí và điều kiện pháp lý trước khi tiếp tục."
                    )
                    age = int(
                        st.number_input(
                            "Tuổi của bạn (Age)", min_value=10, max_value=100, value=18, step=1
                        )
                    )
                    st.write(
                        f"Kiểm tra trước khi gửi live: Giá trị lệnh {draft['notional']:,} • Tổng chi phí ước tính {draft['fee_tax']['total_cost']:,} • Ngoài giờ giao dịch: {'Có' if draft.get('off_session') else 'Không'}"
                    )
                else:
                    age = None

                st.markdown("**Bạn sắp làm gì**")
                st.write(
                    f"- Bạn sẽ {'MUA' if draft['side'] == 'BUY' else ('BÁN' if draft['side'] == 'SELL' else 'MỞ VỊ THẾ BÁN')} (nháp) mã {draft['symbol']}"
                )
                st.write(f"- Khối lượng: {draft['qty']} • Giá dự kiến: {draft['price']}")
                st.write(f"- Ước tính phí/thuế/trượt giá: {draft['fee_tax']['total_cost']}")

                ack1 = st.checkbox(
                    "Tôi hiểu đây không phải lời khuyên đầu tư (Not investment advice)"
                )
                ack2 = st.checkbox("Tôi hiểu có thể thua lỗ (Risk of loss)")
                ack_live = (
                    st.checkbox(
                        "Tôi đủ điều kiện theo quy định và không yêu cầu hướng dẫn lách luật (Eligibility)"
                    )
                    if mode == "live"
                    else False
                )
                if st.button("Xác nhận thực hiện (Confirm execute)", disabled=not api_ready):
                    try:
                        out = api.post(
                            "/simple/confirm_execute",
                            {
                                "portfolio_id": 1,
                                "user_id": str(
                                    st.session_state.get("session_user_id", "streamlit-user")
                                ),
                                "session_id": str(
                                    st.session_state.get("session_id", "streamlit-session")
                                ),
                                "idempotency_token": str(
                                    st.session_state.get("idempotency_token", "")
                                ),
                                "mode": mode,
                                "acknowledged_educational": ack1,
                                "acknowledged_loss": ack2,
                                "acknowledged_live_eligibility": ack_live,
                                "age": age,
                                "draft": draft,
                            },
                        )
                        st.json(out)
                    except Exception as exc:
                        st.error(f"Không thể thực hiện lệnh. Lý do: {exc}")

    with tab_compare:
        compare_market = st.selectbox(
            "Thị trường so sánh (Market)",
            ["Cổ phiếu Việt Nam (VN Stocks)", "Tiền mã hoá (Crypto)"],
            index=0,
            key="compare_market",
        )
        compare_is_crypto = compare_market.startswith("Tiền mã hoá")
        compare_trading_type = "spot_paper"
        compare_exchange = "binance_public"
        if compare_is_crypto:
            compare_trading_type = st.selectbox(
                "Loại giao dịch so sánh (Trading type)",
                ["spot_paper", "perp_paper"],
                index=0,
                key="compare_trading_type",
                format_func=lambda x: (
                    "Giao ngay — giao dịch giấy (Spot paper)"
                    if x == "spot_paper"
                    else "Hợp đồng vĩnh cửu — giao dịch giấy (Perp paper, Long/Short)"
                ),
            )
            compare_exchange = st.selectbox(
                "Sàn dữ liệu so sánh (Exchange)",
                ["binance_public"],
                index=0,
                key="compare_exchange",
                format_func=lambda x: "Binance công khai (Binance public)",
            )
        symbols = st.text_input(
            "Danh sách mã (1 mã hoặc 5–20 mã, phân tách dấu phẩy)",
            value="FPT,VNM,VCB,MWG,HPG",
        )
        lookback = st.slider("Khoảng backtest (mặc định 1 năm / 252 phiên)", 60, 756, 252)
        detail_mode = st.checkbox("Xem chi tiết nâng cao (Advanced)", value=False)
        execution_mode = st.selectbox(
            "Kiểu khớp lệnh (Execution)",
            ["giá đóng cửa (close)", "thanh nến kế tiếp (next-bar)"],
            index=0,
        )
        if st.button("Chạy so sánh (Run comparison)", disabled=not api_ready):
            rows = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if len(rows) != 1 and not (5 <= len(rows) <= 20):
                st.error("Vui lòng nhập đúng 1 mã hoặc từ 5 đến 20 mã để so sánh.")
                return
            resp = api.post(
                "/simple/run_compare",
                {
                    "symbols": rows,
                    "lookback_days": lookback,
                    "timeframe": "1D",
                    "detail_level": "chi tiết" if detail_mode else "tóm tắt",
                    "include_equity_curve": detail_mode,
                    "include_trades": detail_mode,
                    "execution": execution_mode,
                    "market": "crypto" if compare_is_crypto else "vn",
                    "trading_type": compare_trading_type,
                    "exchange": compare_exchange,
                    "include_story_mode": True,
                },
            )
            st.error(resp["warning"])
            if resp.get("story_summary_vi"):
                st.info(resp["story_summary_vi"])
                st.write(resp.get("example_portfolio_vi", ""))
                st.write(resp.get("biggest_drop_vi", ""))

            cards = st.columns(max(1, min(3, len(resp.get("leaderboard", [])))))
            for idx, row in enumerate(resp.get("leaderboard", [])[:3]):
                with cards[idx]:
                    st.markdown(f"**{row.get('model_id', '-')}**")
                    st.write(row.get("example_portfolio_vi", ""))
                    st.write(row.get("biggest_drop_vi", ""))

            with st.expander("Xem chi tiết nâng cao (Advanced)", expanded=False):
                st.dataframe(resp["leaderboard"], use_container_width=True)
            if detail_mode and resp.get("leaderboard"):
                best = resp["leaderboard"][0]
                if best.get("equity_curve"):
                    st.markdown("### Giá trị danh mục theo thời gian (Equity curve)")
                    st.line_chart(best["equity_curve"], x="date", y="nav", use_container_width=True)
                    st.markdown("### Sụt giảm (Drawdown)")
                    st.line_chart(
                        best["equity_curve"], x="date", y="drawdown", use_container_width=True
                    )
                if best.get("trade_list"):
                    st.markdown("### Danh sách giao dịch (Trade list)")
                    st.dataframe(best["trade_list"], use_container_width=True)
                    st.download_button(
                        "Tải CSV giao dịch",
                        data=pd.DataFrame(best["trade_list"]).to_csv(index=False).encode("utf-8"),
                        file_name="trade_list_simple_mode.csv",
                        mime="text/csv",
                    )
            chosen = st.selectbox(
                "Dùng mô hình này cho bước 2 (Use this model)",
                [row["model_id"] for row in resp["leaderboard"]],
            )
            if st.button("Áp dụng lựa chọn mô hình"):
                st.session_state["simple_preferred_model"] = chosen
                st.success(f"Đã lưu lựa chọn: {chosen}. Lưu ý: không tự động giao dịch.")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
