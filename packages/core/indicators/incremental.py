from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from core.indicators import (
    RSIState,
    RollingMeanState,
    ema_incremental,
    rolling_mean_incremental,
    rsi_incremental,
)


@dataclass
class IndicatorState:
    ema20: float | None = None
    ema50: float | None = None
    ema12: float | None = None
    ema26: float | None = None
    macd_signal: float | None = None
    rsi_state: RSIState = field(default_factory=RSIState)
    prev_close: float | None = None
    atr_state: RollingMeanState = field(default_factory=lambda: RollingMeanState.new(14))
    vwap_day: str | None = None
    vwap_pv: float = 0.0
    vwap_v: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "ema20": self.ema20,
            "ema50": self.ema50,
            "ema12": self.ema12,
            "ema26": self.ema26,
            "macd_signal": self.macd_signal,
            "rsi_state": {
                "avg_gain": self.rsi_state.avg_gain,
                "avg_loss": self.rsi_state.avg_loss,
                "prev_close": self.rsi_state.prev_close,
                "warmup_count": self.rsi_state.warmup_count,
            },
            "prev_close": self.prev_close,
            "atr_buf": list(self.atr_state.buf),
            "atr_total": self.atr_state.total,
            "vwap_day": self.vwap_day,
            "vwap_pv": self.vwap_pv,
            "vwap_v": self.vwap_v,
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "IndicatorState":
        st = IndicatorState()
        st.ema20 = data.get("ema20")
        st.ema50 = data.get("ema50")
        st.ema12 = data.get("ema12")
        st.ema26 = data.get("ema26")
        st.macd_signal = data.get("macd_signal")
        rs = data.get("rsi_state", {}) or {}
        st.rsi_state = RSIState(
            avg_gain=float(rs.get("avg_gain", 0.0)),
            avg_loss=float(rs.get("avg_loss", 0.0)),
            prev_close=rs.get("prev_close"),
            warmup_count=int(rs.get("warmup_count", 0)),
        )
        st.prev_close = data.get("prev_close")
        st.atr_state = RollingMeanState.new(14)
        for x in data.get("atr_buf", []) or []:
            st.atr_state.buf.append(float(x))
        st.atr_state.total = float(data.get("atr_total", 0.0))
        st.vwap_day = data.get("vwap_day")
        st.vwap_pv = float(data.get("vwap_pv", 0.0))
        st.vwap_v = float(data.get("vwap_v", 0.0))
        return st


def update_indicators_state(
    state: IndicatorState,
    *,
    end_ts: dt.datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> tuple[dict[str, float], IndicatorState]:
    state.ema20 = ema_incremental(float(close), state.ema20, span=20)
    state.ema50 = ema_incremental(float(close), state.ema50, span=50)
    state.ema12 = ema_incremental(float(close), state.ema12, span=12)
    state.ema26 = ema_incremental(float(close), state.ema26, span=26)
    macd_line = float((state.ema12 or 0.0) - (state.ema26 or 0.0))
    state.macd_signal = ema_incremental(macd_line, state.macd_signal, span=9)
    macd_hist = float(macd_line - (state.macd_signal or 0.0))

    rsi, state.rsi_state = rsi_incremental(float(close), state.rsi_state, window=14)
    if state.rsi_state.avg_loss == 0:
        rsi = 0.0

    prev_close = float(state.prev_close if state.prev_close is not None else open_)
    tr = max(float(high) - float(low), abs(float(high) - prev_close), abs(float(low) - prev_close))
    atr14, state.atr_state = rolling_mean_incremental(tr, state.atr_state)
    state.prev_close = float(close)

    day = end_ts.date().isoformat()
    if state.vwap_day != day:
        state.vwap_day = day
        state.vwap_pv = 0.0
        state.vwap_v = 0.0
    typical = (float(high) + float(low) + float(close)) / 3.0
    state.vwap_pv += typical * float(volume)
    state.vwap_v += float(volume)
    vwap = float(state.vwap_pv / state.vwap_v) if state.vwap_v > 0 else 0.0

    out = {
        "EMA20": float(state.ema20 or 0.0),
        "EMA50": float(state.ema50 or 0.0),
        "RSI14": float(rsi),
        "MACD": float(macd_line),
        "MACD_SIGNAL": float(state.macd_signal or 0.0),
        "MACD_HIST": float(macd_hist),
        "ATR14": float(atr14) if atr14 is not None else 0.0,
        "VWAP": float(vwap),
    }
    return out, state
