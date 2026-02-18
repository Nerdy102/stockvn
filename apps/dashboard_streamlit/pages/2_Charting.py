from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from apps.dashboard_streamlit.ui.cache import cached_get_json, cached_post_json
from apps.dashboard_streamlit.ui.perf import (
    DAILY_MAX_DAYS_DEFAULT,
    INTRADAY_MAX_TRADING_DAYS_DEFAULT,
    MAX_POINTS_PER_CHART,
    downsample_df,
    enforce_bounded_range,
    enforce_intraday_default_window,
)

PAGE_ID = "chart"
PAGE_TITLE = "Chart"


def _tickers_universe() -> list[dict]:
    return cached_get_json("/tickers", params={"limit": 2000, "offset": 0}, ttl_s=1800)


def _to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def render() -> None:
    tickers = _tickers_universe()
    symbols = [t["symbol"] for t in tickers] if tickers else ["FVNA", "FVNB", "VNINDEX"]

    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.selectbox("Symbol", options=symbols, index=0)
    timeframe = c2.selectbox("Timeframe", options=["1D", "1W", "60m", "15m"], index=0)
    adjusted = c3.checkbox("Adjusted (CA/TR aware)", value=True)
    show_ca = c4.checkbox("Show CA markers", value=True)

    end = st.date_input("End", value=st.session_state.get("as_of_date", dt.date(2025, 12, 31)))
    start = st.date_input("Start", value=end - dt.timedelta(days=365))

    tf = "1D" if timeframe == "1W" else timeframe
    start, end = enforce_intraday_default_window(tf, start, end, page_id=PAGE_ID)
    if tf in {"15m", "60m"}:
        st.caption(
            f"Intraday default window bounded to {INTRADAY_MAX_TRADING_DAYS_DEFAULT} trading days."
        )
    enforce_bounded_range(start, end, DAILY_MAX_DAYS_DEFAULT, page_id=PAGE_ID)

    ohlcv_resp = cached_get_json(
        "/chart/ohlcv",
        params={
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjusted": adjusted,
        },
        ttl_s=300 if tf == "1D" else 60,
    )
    ind_resp = cached_get_json(
        "/chart/indicators",
        params={
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjusted": adjusted,
            "indicators": "SMA20,EMA20,RSI14,MACD,ATR14,VWAP",
        },
        ttl_s=300 if tf == "1D" else 60,
    )
    alpha_resp = cached_get_json(
        "/chart/alpha",
        params={
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "model_id": "alpha_v3",
            "timeframe": tf,
        },
        ttl_s=300,
    )

    ohlcv = _to_df(ohlcv_resp.get("rows", []))
    indicators = _to_df(ind_resp.get("rows", []))
    alpha = _to_df(alpha_resp.get("rows", []))

    if ohlcv.empty:
        st.warning("No OHLCV data.")
        return

    if len(ohlcv) > MAX_POINTS_PER_CHART:
        ohlcv = downsample_df(ohlcv, MAX_POINTS_PER_CHART, page_id=PAGE_ID)
    if not indicators.empty and len(indicators) > MAX_POINTS_PER_CHART:
        indicators = downsample_df(indicators, MAX_POINTS_PER_CHART, page_id=PAGE_ID)
    if not alpha.empty and len(alpha) > MAX_POINTS_PER_CHART:
        alpha = downsample_df(alpha, MAX_POINTS_PER_CHART, page_id=PAGE_ID)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=["Price + Volume", "Indicators", "Alpha / Uncertainty"],
    )

    fig.add_trace(
        go.Candlestick(
            x=ohlcv["timestamp"],
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            name="Price",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=ohlcv["timestamp"], y=ohlcv["volume"], name="Volume", opacity=0.3),
        row=1,
        col=1,
        secondary_y=True,
    )

    if not indicators.empty:
        for col, color in [("SMA20", "#22c55e"), ("EMA20", "#f59e0b"), ("VWAP", "#8b5cf6")]:
            if col in indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicators["timestamp"],
                        y=indicators[col],
                        mode="lines",
                        name=col,
                        line={"color": color},
                    ),
                    row=1,
                    col=1,
                    secondary_y=False,
                )

        if "RSI14" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators["timestamp"], y=indicators["RSI14"], mode="lines", name="RSI14"
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="#ef4444")
            fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="#22c55e")

        if "MACD" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators["timestamp"], y=indicators["MACD"], mode="lines", name="MACD"
                ),
                row=2,
                col=1,
            )
        if "MACD_SIGNAL" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators["timestamp"],
                    y=indicators["MACD_SIGNAL"],
                    mode="lines",
                    name="MACD_SIGNAL",
                ),
                row=2,
                col=1,
            )
        if "MACD_HIST" in indicators.columns:
            fig.add_trace(
                go.Bar(
                    x=indicators["timestamp"],
                    y=indicators["MACD_HIST"],
                    name="MACD_HIST",
                    opacity=0.25,
                ),
                row=2,
                col=1,
            )

    if show_ca:
        for ca in ohlcv_resp.get("ca_markers", []):
            x = pd.to_datetime(ca["ex_date"])
            fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="#94a3b8", row=1, col=1)

    if not alpha.empty:
        xcol = "date" if "date" in alpha.columns else "timestamp"
        fig.add_trace(
            go.Scatter(
                x=alpha[xcol],
                y=alpha.get("mu_norm", pd.Series(dtype=float)),
                mode="lines",
                name="alpha_mu_norm",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=alpha[xcol],
                y=alpha.get("lo_norm", pd.Series(dtype=float)),
                mode="lines",
                name="alpha_lo_norm",
                line={"dash": "dot"},
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=alpha[xcol],
                y=alpha.get("hi_norm", pd.Series(dtype=float)),
                mode="lines",
                name="alpha_hi_norm",
                line={"dash": "dot"},
                fill="tonexty",
                opacity=0.2,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

    fig.update_layout(height=980, xaxis_rangeslider_visible=False, legend_orientation="h")
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI/MACD", row=2, col=1)
    fig.update_yaxes(title_text="Alpha Norm", row=3, col=1, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Persistent annotations")
    workspaces = cached_get_json("/workspaces", params={"user_id": ""}, ttl_s=300)
    if not workspaces:
        st.info("No workspace found. Create one in Watchlists page.")
        return
    ws = st.selectbox(
        "Workspace", options=workspaces, format_func=lambda x: x["name"], key="chart_ws"
    )
    annotations = cached_get_json(
        "/chart/annotations",
        params={"workspace_id": ws["id"], "symbol": symbol, "timeframe": timeframe},
        ttl_s=60,
    )
    versions = annotations.get("versions", [])
    if versions:
        ver_choice = st.selectbox(
            "Saved version",
            options=versions,
            format_func=lambda x: f"v{x['version']} ({x['updated_at']})",
        )
        st.json(ver_choice.get("shapes_json", {}))
    else:
        st.caption("No saved annotation yet.")

    default_shapes = versions[0].get("shapes_json", {}).get("shapes", []) if versions else []
    shapes_text = st.text_area(
        "Shapes JSON (line/rect only)",
        value=json.dumps(default_shapes, ensure_ascii=False, indent=2),
        height=180,
    )
    notes = st.text_input("Audit note", value="manual save")

    csave, cundo = st.columns(2)
    with csave:
        if st.button("Save annotations"):
            try:
                shapes = json.loads(shapes_text)
                if not isinstance(shapes, list):
                    raise ValueError("shapes_json must be list")
                res = cached_post_json(
                    "/chart/annotations",
                    payload={
                        "workspace_id": ws["id"],
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "window_start": start.isoformat(),
                        "window_end": end.isoformat(),
                        "actor": "streamlit_user",
                        "notes": notes,
                        "shapes_json": shapes,
                    },
                    ttl_s=1,
                )
                st.success(f"Saved version {res['version']}")
            except Exception as exc:
                st.error(f"Invalid shapes or save failed: {exc}")
    with cundo:
        if st.button("Undo to previous"):
            if len(versions) >= 2:
                prev = versions[1].get("shapes_json", {}).get("shapes", [])
                st.session_state["chart_shapes_override"] = json.dumps(
                    prev, ensure_ascii=False, indent=2
                )
                st.success("Loaded previous version into editor.")
            else:
                st.warning("No previous version to undo.")

    if "chart_shapes_override" in st.session_state:
        st.code(st.session_state["chart_shapes_override"], language="json")
