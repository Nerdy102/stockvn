from __future__ import annotations

import httpx
import streamlit as st

from apps.dashboard_streamlit.lib import api

PAGE_ID = "home_dashboard"
PAGE_TITLE = "🏠 Tổng quan hôm nay"
FONT_STACK_VI = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'


@st.cache_data(ttl=120)
def _load_dashboard(
    universe: str,
    timeframe: str,
    limit_signals: int,
    lookback_sessions: int,
    market: str,
    trading_type: str,
    exchange: str,
) -> dict:
    return api.get(
        "/simple/dashboard",
        {
            "universe": universe,
            "timeframe": timeframe,
            "limit_signals": limit_signals,
            "lookback_sessions": lookback_sessions,
            "market": market,
            "trading_type": trading_type,
            "exchange": exchange,
        },
    )


def _go_simple_mode(symbol: str, model_id: str, timeframe: str) -> None:
    st.session_state["simple_prefill_symbol"] = symbol
    st.session_state["simple_preferred_model"] = model_id
    st.session_state["simple_prefill_timeframe"] = timeframe
    st.success("Đã lưu cấu hình. Vui lòng mở trang 🚀 Giao dịch đơn giản (Simple mode).")


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
    st.title("🏠 Tổng quan hôm nay (Today dashboard)")
    st.info(
        "Kiểm tra hiển thị dấu: Tôi hiểu đây là công cụ giáo dục, không phải lời khuyên đầu tư."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        market_view = st.selectbox(
            "Xem thị trường",
            ["vn", "crypto", "both"],
            index=0,
            format_func=lambda x: {
                "vn": "Cổ phiếu Việt Nam (VN Stocks)",
                "crypto": "Tiền mã hoá (Crypto)",
                "both": "Cả hai (Both)",
            }[x],
        )
    with c2:
        universe = st.selectbox("Rổ cổ phiếu (Universe)", ["VN30", "VNINDEX", "ALL"], index=0)
    with c3:
        timeframe = st.selectbox("Khung thời gian (Timeframe)", ["1D", "60m"], index=0)
    with c4:
        limit_signals = st.slider("Giới hạn tín hiệu (Signal limit)", 5, 20, 10)
    with c5:
        lookback = st.slider("Số phiên kiểm chứng (Backtest sessions)", 60, 756, 252)
    with c6:
        trading_type = st.selectbox(
            "Loại giao dịch Crypto",
            ["spot_paper", "perp_paper"],
            index=0,
            format_func=lambda x: (
                "Giao ngay — giao dịch giấy (Spot paper)"
                if x == "spot_paper"
                else "Hợp đồng vĩnh cửu — giao dịch giấy (Perp paper, Long/Short)"
            ),
        )
    exchange = st.selectbox(
        "Sàn dữ liệu Crypto",
        ["binance_public"],
        index=0,
        format_func=lambda x: "Binance công khai (Binance public)",
    )

    if st.button("Đồng bộ dữ liệu (Sync data)"):
        try:
            out = api.post("/simple/dashboard/refresh", {})
            st.success(out.get("message", "Đã đồng bộ"))
            _load_dashboard.clear()
        except (httpx.HTTPError, ValueError):
            st.warning("Không thể đồng bộ vì API chưa sẵn sàng.")

    try:
        data = _load_dashboard(
            universe, timeframe, limit_signals, lookback, market_view, trading_type, exchange
        )
    except (httpx.HTTPError, ValueError):
        st.warning(
            "Chưa kết nối được API tổng quan. Hãy chạy API hoặc dùng verify-offline để kiểm tra."
        )
        return

    st.caption(f"Ngày dữ liệu mới nhất (Latest data date): {data.get('as_of_date', 'N/A')}")

    st.subheader("Tình hình thị trường hôm nay (Market today)")
    market = data.get("market_summary", data.get("market_today_summary", {}))
    st.write(market.get("text", "Chưa có dữ liệu tóm tắt."))

    st.subheader("Tín hiệu đáng chú ý (Research signals) — MUA/BÁN (nháp)")
    t_buy, t_sell = st.tabs(
        [
            "Ứng viên MUA (nháp) (Draft BUY candidates)",
            "Ứng viên BÁN (nháp) (Draft SELL candidates)",
        ]
    )
    with t_buy:
        buys = data.get("buy_candidates", data.get("signals_buy_candidates", []))
        if not buys:
            st.warning("Chưa có ứng viên MUA (nháp) phù hợp.")
        for i, row in enumerate(buys[:20]):
            with st.container(border=True):
                st.write(
                    f"**{row['symbol']}** • {row['model']} • Tín hiệu: {row['signal']} • Độ tin cậy: {row['confidence']}"
                )
                st.write(f"Lý do ngắn: {row['reason']}")
                st.write("Rủi ro: " + ", ".join(row.get("risks", [])))
                if st.button(
                    f"Mở chế độ đơn giản (Open Simple Mode) #{i+1}",
                    key=f"open_simple_buy_{i}",
                ):
                    _go_simple_mode(row["symbol"], row["model_id"], timeframe)

    with t_sell:
        sells = data.get("sell_candidates", data.get("signals_sell_candidates", []))
        if not sells:
            st.warning("Chưa có ứng viên BÁN (nháp) phù hợp.")
        for i, row in enumerate(sells[:20]):
            with st.container(border=True):
                st.write(
                    f"**{row['symbol']}** • {row['model']} • Tín hiệu: {row['signal']} • Độ tin cậy: {row['confidence']}"
                )
                st.write(f"Lý do ngắn: {row['reason']}")
                st.write("Rủi ro: " + ", ".join(row.get("risks", [])))
                if st.button(
                    f"Mở chế độ đơn giản (Open Simple Mode) BÁN #{i+1}",
                    key=f"open_simple_sell_{i}",
                ):
                    _go_simple_mode(row["symbol"], row["model_id"], timeframe)

    st.subheader("Hiệu quả mô hình (Model performance)")
    st.error(
        "CẢNH BÁO (Warning): Quá khứ không đảm bảo tương lai (Past performance is not indicative of future results); có rủi ro overfit; chi phí thực tế có thể khác mô phỏng."
    )
    perf = data.get("model_leaderboard", data.get("model_performance_leaderboard", []))
    if perf:
        st.dataframe(perf, use_container_width=True)
        st.caption(f"ID báo cáo (Report ID): {perf[0].get('report_id','N/A')}")

    st.subheader("Danh mục giao dịch giấy (Paper portfolio)")
    p = data.get("paper_portfolio_summary", {})
    st.write(
        f"Trạng thái: {p.get('message','')} • Lãi/lỗ tạm tính (P&L): {p.get('pnl',0):,.0f} • Số lệnh: {p.get('total_orders',0)} • Số mã nắm giữ: {p.get('open_positions',0)} • Tỷ lệ tiền mặt: {p.get('cash_ratio',0):.2f}"
    )
    if p.get("top_positions"):
        st.dataframe(p["top_positions"], use_container_width=True)


    st.subheader("Trạng thái hệ thống (System health)")
    sys_status = data.get("system_status", {})
    st.write(
        f"Môi trường giao dịch: {sys_status.get('trading_env','N/A')} • Trạng thái giao dịch thật (Live status): {sys_status.get('live_status','TẮT')} • Kill-switch: {sys_status.get('kill_switch','N/A')} • Kết nối broker: {sys_status.get('broker_connectivity','N/A')}"
    )
    if sys_status.get("live_block_reason"):
        st.error(f"Lý do bị chặn (Block reason): {sys_status.get('live_block_reason')}")

    st.subheader("Trạng thái dữ liệu (Data status)")
    d = data.get("data_status", {})
    st.write(
        f"Nhà cung cấp dữ liệu (Provider): {d.get('provider','N/A')} • Số mã: {d.get('symbols_count',0)} • Số dòng dữ liệu: {d.get('rows',0)} • Khung thời gian sẵn có: {', '.join(d.get('timeframes', []))} • Lần cập nhật gần nhất (Last update): {d.get('last_update','N/A')}"
    )

    st.subheader("Cảnh báo rủi ro (Risk disclaimers)")
    for txt in data.get("disclaimers", []):
        st.write(f"- {txt}")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
