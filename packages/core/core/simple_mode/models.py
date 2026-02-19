from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.simple_mode.schemas import SignalResult


@dataclass
class ModelProfile:
    model_id: str
    title: str
    description: str


MODEL_PROFILES = [
    ModelProfile("model_1", "Model 1 — Xu hướng", "EMA20/EMA50 + breakout + volume"),
    ModelProfile("model_2", "Model 2 — Hồi quy về trung bình", "RSI14 + khoảng cách EMA20 + ATR%"),
    ModelProfile("model_3", "Model 3 — Factor + Regime", "Đa yếu tố + lọc regime, ít giao dịch"),
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


def run_signal(model_id: str, symbol: str, timeframe: str, df: pd.DataFrame) -> SignalResult:
    if df.empty:
        return SignalResult(
            symbol=symbol,
            timeframe=timeframe,
            model_id=model_id,
            as_of=dt.datetime.utcnow(),
            signal="UU_TIEN_QUAN_SAT",
            confidence="THAP",
            explanation=["Thiếu dữ liệu giá để chạy mô hình."],
            risks=["Thiếu dữ liệu"],
        )
    w = df.copy().reset_index(drop=True)
    w["ema20"] = _ema(w["close"], 20)
    w["ema50"] = _ema(w["close"], 50)
    w["rsi14"] = _rsi(w["close"], 14)
    w["atr14"] = _atr(w, 14)
    last = w.iloc[-1]

    if model_id == "model_1":
        hh20 = float(w["high"].tail(20).max())
        vol20 = float(w["volume"].tail(20).mean()) if len(w) >= 20 else float(w["volume"].mean())
        trend = float(last["ema20"]) > float(last["ema50"]) and float(last["close"]) > float(
            last["ema50"]
        )
        breakout = float(last["close"]) >= hh20 and float(last["volume"]) > 1.5 * max(vol20, 1.0)
        signal = (
            "TANG"
            if trend and breakout
            else ("GIAM" if float(last["close"]) < float(last["ema50"]) else "TRUNG_TINH")
        )
        side = "BUY" if signal == "TANG" else ("SELL" if signal == "GIAM" else "HOLD")
        conf = "CAO" if signal in {"TANG", "GIAM"} else "VUA"
        explanation = [
            "Mô hình xu hướng dùng EMA20/EMA50 và breakout 20 phiên.",
            f"EMA20={last['ema20']:.2f}, EMA50={last['ema50']:.2f}, close={last['close']:.2f}.",
            "Tín hiệu là nghiên cứu định lượng, không phải khuyến nghị đầu tư.",
        ]
    elif model_id == "model_2":
        dist = (float(last["close"]) - float(last["ema20"])) / max(float(last["ema20"]), 1e-9)
        atr_pct = float(last["atr14"]) / max(float(last["close"]), 1e-9)
        rsi = float(last["rsi14"]) if not np.isnan(float(last["rsi14"])) else 50.0
        if rsi < 30 and dist < -0.01 and atr_pct < 0.08:
            signal, side = "TANG", "BUY"
        elif rsi > 70 and dist > 0.01:
            signal, side = "GIAM", "SELL"
        else:
            signal, side = "TRUNG_TINH", "HOLD"
        conf = "VUA" if signal != "TRUNG_TINH" else "THAP"
        explanation = [
            "Mô hình mean-reversion dùng RSI14 + khoảng cách so với EMA20 + ATR%.",
            f"RSI14={rsi:.2f}, distance_to_EMA20={dist:.4f}, ATR%={atr_pct:.4f}.",
            "Mô hình giao dịch dày hơn nên cần chú ý phí/slippage.",
        ]
    else:
        ret20 = float(w["close"].pct_change(20).iloc[-1] if len(w) > 21 else 0.0)
        vol20 = float(w["close"].pct_change().tail(20).std() or 0.0)
        regime_risk_off = float(last["close"]) < float(last["ema50"])
        score = 0.6 * ret20 - 0.4 * vol20
        if regime_risk_off:
            signal, side = "TRUNG_TINH", "HOLD"
        elif score > 0.02:
            signal, side = "TANG", "BUY"
        elif score < -0.02:
            signal, side = "GIAM", "SELL"
        else:
            signal, side = "UU_TIEN_QUAN_SAT", "HOLD"
        conf = "CAO" if (not regime_risk_off and signal != "UU_TIEN_QUAN_SAT") else "THAP"
        explanation = [
            "Mô hình đa yếu tố (value/quality/momentum/low-vol/dividend) + lọc regime.",
            f"Factor score xấp xỉ={score:.4f}, regime_risk_off={regime_risk_off}.",
            "Ưu tiên giảm giao dịch khi regime xấu.",
        ]

    risks = ["Kết quả quá khứ không đảm bảo tương lai", "Rủi ro thanh khoản và trượt giá"]
    return SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        model_id=model_id,
        as_of=dt.datetime.utcnow(),
        signal=signal,  # type: ignore[arg-type]
        confidence=conf,  # type: ignore[arg-type]
        proposed_side=side,  # type: ignore[arg-type]
        explanation=explanation,
        risks=risks,
        indicators={
            "ema20": float(last["ema20"]),
            "ema50": float(last["ema50"]),
            "rsi14": float(last["rsi14"]) if not np.isnan(float(last["rsi14"])) else 50.0,
            "atr14": float(last["atr14"]) if not np.isnan(float(last["atr14"])) else 0.0,
        },
        latest_price=float(last["close"]),
        marker_time=str(w.iloc[-1].get("date") or w.iloc[-1].get("timestamp")),
    )
