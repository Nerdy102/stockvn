from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from data.quality.gates import MIN_ROWS_1D, MIN_ROWS_60M, run_quality_gates

from core.simple_mode.confidence import (
    build_risk_tags,
    compact_debug_fields,
    confidence_bucket,
    detect_regime,
    liquidity_bucket,
)
from core.simple_mode.schemas import SignalResult


@dataclass
class ModelProfile:
    model_id: str
    title: str
    description: str


MODEL_PROFILES = [
    ModelProfile("model_1", "Mô hình 1 — Xu hướng", "EMA20/EMA50 + bứt phá + khối lượng"),
    ModelProfile("model_2", "Mô hình 2 — Hồi quy trung bình", "RSI14 + khoảng cách EMA20 + ATR%"),
    ModelProfile("model_3", "Mô hình 3 — Đa yếu tố + chế độ", "Đa yếu tố + lọc chế độ"),
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _limit_sentence(text: str, max_len: int = 140) -> str:
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"


def _neutral_signal(
    symbol: str, timeframe: str, model_id: str, reason: str, failures: list[str]
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        model_id=model_id,
        as_of=dt.datetime.utcnow(),
        signal="TRUNG_TINH",
        confidence="THAP",
        proposed_side="HOLD",
        explanation=[reason],
        reason_short=_limit_sentence(reason),
        reason_bullets=[reason, f"Lỗi dữ liệu: {', '.join(failures[:2])}" if failures else ""],
        risk_tags=["Thiếu dữ liệu"],
        confidence_bucket="Thấp",
        risks=["Thiếu dữ liệu"],
        debug_fields={"gate_failures": ",".join(failures)},
    )


def run_signal(model_id: str, symbol: str, timeframe: str, df: pd.DataFrame) -> SignalResult:
    if df.empty:
        return _neutral_signal(
            symbol,
            timeframe,
            model_id,
            "Thiếu dữ liệu để phân tích (cần tối thiểu phiên).",
            ["EMPTY_DATA"],
        )

    gate = run_quality_gates(df, timeframe)
    min_rows = MIN_ROWS_60M if timeframe == "60m" else MIN_ROWS_1D
    if not gate.ok:
        return _neutral_signal(
            symbol,
            timeframe,
            model_id,
            f"Thiếu dữ liệu để phân tích (cần tối thiểu {min_rows} phiên).",
            gate.failures,
        )

    w = gate.cleaned.copy().reset_index(drop=True)
    w["ema20"] = _ema(w["close"], 20)
    w["ema50"] = _ema(w["close"], 50)
    w["rsi14"] = _rsi(w["close"], 14)
    w["atr14"] = _atr(w, 14)
    w["vol_avg20"] = w["volume"].rolling(20).mean()
    w["high20_prev"] = w["high"].rolling(20).max().shift(1)

    last = w.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"]) if not np.isnan(float(last["rsi14"])) else 50.0
    atr14 = float(last["atr14"]) if not np.isnan(float(last["atr14"])) else 0.0
    atr_pct = atr14 / max(close, 1e-9)
    vol = float(last["volume"])
    vol_avg20 = float(last["vol_avg20"]) if not np.isnan(float(last["vol_avg20"])) else vol
    high20_prev = float(last["high20_prev"]) if not np.isnan(float(last["high20_prev"])) else close

    regime = detect_regime(close, ema20, ema50)
    liq_bucket = liquidity_bucket(close, vol, (w["close"] * w["volume"]).tail(60))

    if model_id == "model_1":
        trend_bias = ema20 > ema50 and close > ema50
        entry = close > high20_prev and vol > 1.5 * max(vol_avg20, 1.0) and atr_pct >= 0.01
        exit_cond = close < ema50
        signal = "TANG" if (trend_bias and entry) else ("GIAM" if exit_cond else "TRUNG_TINH")
        side = "BUY" if signal == "TANG" else ("SELL" if signal == "GIAM" else "HOLD")
        reason_short = (
            "Giá vượt đỉnh 20 phiên trước và khối lượng tăng."
            if signal == "TANG"
            else (
                "Giá rơi dưới EMA50, ưu tiên hạ rủi ro."
                if signal == "GIAM"
                else "Xu hướng chưa đủ rõ để vào lệnh mới."
            )
        )
        regime_expected = "risk_on"
    elif model_id == "model_2":
        oversold = rsi14 < 30 and close < ema20 and atr_pct >= 0.008
        overbought = rsi14 > 70 and close > ema20 and atr_pct >= 0.008
        exit_cond = (45 <= rsi14 <= 55) or (abs(close - ema20) / max(close, 1e-9) < 0.003)
        signal = (
            "TANG"
            if oversold
            else ("GIAM" if overbought else ("TRUNG_TINH" if exit_cond else "TRUNG_TINH"))
        )
        side = "BUY" if signal == "TANG" else ("SELL" if signal == "GIAM" else "HOLD")
        reason_short = (
            "RSI thấp và giá dưới EMA20, có thể hồi kỹ thuật."
            if signal == "TANG"
            else (
                "RSI cao và giá vượt EMA20, dễ điều chỉnh."
                if signal == "GIAM"
                else "Động lượng hồi quy trung tính, ưu tiên quan sát."
            )
        )
        regime_expected = "risk_on"
    else:
        if regime == "risk_off":
            signal, side = "TRUNG_TINH", "HOLD"
            reason_short = "Chế độ rủi ro-xấu, ưu tiên quan sát."
        else:
            ret20 = float(w["close"].pct_change(20).iloc[-1] if len(w) > 21 else 0.0)
            smooth_rank = float(pd.Series([ret20]).ewm(span=5, adjust=False).mean().iloc[-1])
            signal = "TANG" if (smooth_rank >= 0.0 and 0.008 <= atr_pct <= 0.06) else "TRUNG_TINH"
            side = "BUY" if signal == "TANG" else "HOLD"
            reason_short = (
                "Điểm xu hướng đã làm mượt đang ở vùng thuận lợi."
                if signal == "TANG"
                else "Điểm yếu tố chưa đủ mạnh để phát lệnh."
            )
        regime_expected = "risk_on"

    conf_bucket, _score = confidence_bucket(
        has_min_rows=True,
        liquidity=liq_bucket,
        regime=regime,
        regime_expected=regime_expected,
        atr_pct=atr_pct,
    )
    conf = "CAO" if conf_bucket == "Cao" else "VUA" if conf_bucket == "Vừa" else "THAP"
    risk_tags = build_risk_tags(
        liquidity=liq_bucket,
        regime=regime,
        atr_pct=atr_pct,
        has_min_rows=True,
    )
    explanation = [
        reason_short,
        f"Chế độ thị trường: {regime}; thanh khoản: {liq_bucket}; ATR%={atr_pct*100:.2f}%.",
        "Đây là tín hiệu nghiên cứu phục vụ giáo dục, không phải khuyến nghị đầu tư.",
    ]
    debug_fields = compact_debug_fields(
        {
            "ema20": ema20,
            "ema50": ema50,
            "rsi14": rsi14,
            "atr14": atr14,
            "atr_pct": atr_pct,
            "vol": vol,
            "vol_avg20": vol_avg20,
            "high20_prev": high20_prev,
            "close": close,
        }
    )

    return SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        model_id=model_id,
        as_of=dt.datetime.utcnow(),
        signal=signal,  # type: ignore[arg-type]
        confidence=conf,  # type: ignore[arg-type]
        proposed_side=side,  # type: ignore[arg-type]
        explanation=explanation,
        reason_short=_limit_sentence(reason_short),
        reason_bullets=explanation[:3],
        risk_tags=risk_tags[:2],
        confidence_bucket=conf_bucket,  # type: ignore[arg-type]
        risks=["Quá khứ không đảm bảo tương lai", "Rủi ro thanh khoản và trượt giá"],
        indicators={
            "ema20": ema20,
            "ema50": ema50,
            "rsi14": rsi14,
            "atr14": atr14,
        },
        debug_fields=debug_fields,
        latest_price=close,
        marker_time=str(last.get("date") or last.get("timestamp")),
    )
