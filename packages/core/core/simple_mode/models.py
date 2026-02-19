from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import Session, SQLModel

from core.db.session import get_database_url, get_engine
from core.features.compute_features import compute_features
from core.simple_mode.audit_models import SignalAudit
from core.simple_mode.confidence import compact_debug_fields
from core.simple_mode.confidence_v2 import compute_confidence_v2, detect_regime
from core.simple_mode.schemas import SignalResult
from data.quality.gates import MIN_ROWS_1D, MIN_ROWS_60M, dataset_hash, run_quality_gates
from data.quality.models import DataQualityEvent


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


def _limit_sentence(text: str, max_len: int = 140) -> str:
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"


def _hash_obj(v: object) -> str:
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _code_hash() -> str:
    code = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _persist_quality_events(*, market: str, symbol: str, timeframe: str, dataset_h: str, failures: list[str], warnings: list[str]) -> None:
    if not failures and not warnings:
        return
    try:
        engine = get_engine(get_database_url())
        SQLModel.metadata.create_all(engine, tables=[DataQualityEvent.__table__])
        with Session(engine) as session:
            for code in failures:
                session.add(DataQualityEvent(market=market, symbol=symbol, timeframe=timeframe, severity="error", code=code, message=f"Gate thất bại: {code}", dataset_hash=dataset_h))
            for code in warnings:
                session.add(DataQualityEvent(market=market, symbol=symbol, timeframe=timeframe, severity="warning", code=code, message=f"Cảnh báo dữ liệu: {code}", dataset_hash=dataset_h))
            session.commit()
    except Exception:
        return


def _persist_signal_audit(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    model_id: str,
    signal: str,
    confidence_bucket: str,
    confidence_score: int,
    reason_short: str,
    risk_tags: list[str],
    debug_fields: dict[str, object],
    config_hash: str,
    dataset_h: str,
    code_hash: str,
) -> None:
    try:
        engine = get_engine(get_database_url())
        SQLModel.metadata.create_all(engine, tables=[SignalAudit.__table__])
        with Session(engine) as session:
            session.add(
                SignalAudit(
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_id=model_id,
                    signal=signal,
                    confidence_bucket=confidence_bucket,
                    confidence_score=confidence_score,
                    reason_short=_limit_sentence(reason_short),
                    risk_tags_json=json.dumps(risk_tags[:2], ensure_ascii=False),
                    debug_fields_json=json.dumps(debug_fields, ensure_ascii=False, default=str),
                    config_hash=config_hash,
                    dataset_hash=dataset_h,
                    code_hash=code_hash,
                )
            )
            session.commit()
    except Exception:
        return


def _neutral_signal(symbol: str, timeframe: str, model_id: str, reason: str, failures: list[str], *, risk_tag: str = "Thiếu dữ liệu", debug_fields: dict[str, object] | None = None) -> SignalResult:
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
        reason_bullets=[reason],
        risk_tags=[risk_tag],
        confidence_bucket="Thấp",
        risks=[risk_tag],
        debug_fields=(debug_fields or {}),
    )


def run_signal(model_id: str, symbol: str, timeframe: str, df: pd.DataFrame, *, market: str = "vn", as_of_ts: object | None = None) -> SignalResult:
    config_hash = _hash_obj({"model_id": model_id, "timeframe": timeframe, "market": market})
    code_hash = _code_hash()
    start = str(df.iloc[0].get("date") if not df.empty else "")
    end = str(df.iloc[-1].get("date") if not df.empty else "")
    dataset_h = dataset_hash(market=market, symbol=symbol, timeframe=timeframe, start=start, end=end, df=df) if not df.empty else _hash_obj({"empty": True, "symbol": symbol})

    if df.empty:
        reason = "Thiếu/không hợp lệ dữ liệu để phân tích (mã lỗi: EMPTY_DATA)."
        out = _neutral_signal(symbol, timeframe, model_id, reason, ["EMPTY_DATA"], debug_fields={"gate_failures": ["EMPTY_DATA"], "config_hash": config_hash, "dataset_hash": dataset_h, "code_hash": code_hash})
        _persist_signal_audit(market=market, symbol=symbol, timeframe=timeframe, model_id=model_id, signal=out.signal, confidence_bucket=out.confidence_bucket, confidence_score=0, reason_short=out.reason_short, risk_tags=out.risk_tags, debug_fields=out.debug_fields, config_hash=config_hash, dataset_h=dataset_h, code_hash=code_hash)
        return out

    gate = run_quality_gates(df, timeframe, market=market)
    _persist_quality_events(market=market, symbol=symbol, timeframe=timeframe, dataset_h=dataset_h, failures=gate.failures, warnings=gate.warnings)
    min_rows = MIN_ROWS_60M if timeframe == "60m" else MIN_ROWS_1D
    if not gate.ok:
        reason = f"Thiếu/không hợp lệ dữ liệu để phân tích (mã lỗi: {','.join(gate.failures[:2])})."
        risk_tag = "Thiếu dữ liệu" if any("MIN_ROWS" in x for x in gate.failures) else "Dữ liệu không hợp lệ"
        out = _neutral_signal(symbol, timeframe, model_id, reason, gate.failures, risk_tag=risk_tag, debug_fields={"gate_failures": gate.failures, "degraded_ok": True, "required_min_rows": min_rows, "config_hash": config_hash, "dataset_hash": dataset_h, "code_hash": code_hash})
        _persist_signal_audit(market=market, symbol=symbol, timeframe=timeframe, model_id=model_id, signal=out.signal, confidence_bucket=out.confidence_bucket, confidence_score=0, reason_short=out.reason_short, risk_tags=out.risk_tags, debug_fields=out.debug_fields, config_hash=config_hash, dataset_h=dataset_h, code_hash=code_hash)
        return out

    w = compute_features(gate.cleaned, as_of_ts=as_of_ts)
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
    low20_prev = float(last["low20_prev"]) if not np.isnan(float(last["low20_prev"])) else close
    regime = detect_regime(close, ema20, ema50)

    if model_id == "model_1":
        bias_up = (ema20 > ema50) and (close > ema50)
        entry_up = bool(last["breakout_up"]) and (vol > 1.5 * max(vol_avg20, 1.0)) and (atr_pct >= 0.005)
        exit_cond = close < ema50
        signal = "TANG" if (bias_up and entry_up) else ("GIAM" if exit_cond else "TRUNG_TINH")
        side = "BUY" if signal == "TANG" else ("SELL" if signal == "GIAM" else "HOLD")
        reason_short = "Mô hình xu hướng chưa hội tụ điều kiện rõ ràng." if signal == "TRUNG_TINH" else ("Giá bứt phá qua đỉnh 20 phiên trước cùng khối lượng mạnh." if signal == "TANG" else "Giá rơi xuống dưới EMA50, ưu tiên giảm rủi ro.")
    elif model_id == "model_2":
        oversold = (rsi14 < 30) and (close < ema20) and (atr_pct >= 0.004)
        overbought = (rsi14 > 70) and (close > ema20) and (atr_pct >= 0.004)
        signal = "TANG" if oversold else ("GIAM" if overbought else "TRUNG_TINH")
        side = "BUY" if signal == "TANG" else ("SELL" if signal == "GIAM" else "HOLD")
        reason_short = "Tín hiệu hồi quy chưa đủ mạnh, tạm thời trung tính." if signal == "TRUNG_TINH" else ("RSI quá bán, có xác suất hồi kỹ thuật." if signal == "TANG" else "RSI quá mua, rủi ro điều chỉnh tăng cao.")
    else:
        if regime == "risk_off":
            signal, side = "TRUNG_TINH", "HOLD"
            reason_short = "Chế độ rủi ro-xấu (risk-off), buộc giữ trung tính."
        else:
            factor_percentile = float(last.get("factor_percentile", 0.5))
            if not (0.003 <= atr_pct <= 0.06):
                signal = "TRUNG_TINH"
            elif factor_percentile >= 0.8:
                signal = "TANG"
            elif factor_percentile <= 0.2:
                signal = "GIAM"
            else:
                signal = "TRUNG_TINH"
            side = "BUY" if signal == "TANG" else ("SELL" if signal == "GIAM" else "HOLD")
            reason_short = "Mô hình nhân tố chưa cho điểm cực trị để phát lệnh." if signal == "TRUNG_TINH" else ("Điểm nhân tố ở nhóm cao, ưu tiên xu hướng tăng." if signal == "TANG" else "Điểm nhân tố ở nhóm thấp, cảnh báo suy yếu.")

    total_score, conf_bucket, conf_tags, score_debug = compute_confidence_v2(has_min_rows=True, close=close, volume=vol, dollar_vol_lookback=(w["close"] * w["volume"]).tail(60), atr_pct=atr_pct, model_id=model_id, regime=regime)
    conf = "CAO" if conf_bucket == "Cao" else "VUA" if conf_bucket == "Vừa" else "THAP"

    risk_tags: list[str] = []
    if len(w) < min_rows:
        risk_tags.append("Thiếu dữ liệu")
    if any(conf_tags):
        risk_tags.extend(conf_tags)
    if regime == "risk_off":
        risk_tags.append("Chế độ rủi ro-xấu (risk-off)")
    if gate.outlier_jump_flagged:
        risk_tags.append("Dữ liệu bất thường")
    seen = set()
    risk_tags = [x for x in risk_tags if not (x in seen or seen.add(x))][:2]

    debug_fields = compact_debug_fields({
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi14,
        "atr14": atr14,
        "atr_pct": atr_pct,
        "vol": vol,
        "vol_avg20": vol_avg20,
        "high20_prev": high20_prev,
        "low20_prev": low20_prev,
        "close": close,
        "gate_failures": gate.failures,
        "config_hash": config_hash,
        "dataset_hash": dataset_h,
        "code_hash": code_hash,
        **score_debug,
    })

    out = SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        model_id=model_id,
        as_of=dt.datetime.utcnow(),
        signal=signal,
        confidence=conf,
        proposed_side=side,
        explanation=[reason_short],
        reason_short=_limit_sentence(reason_short),
        reason_bullets=[reason_short],
        risk_tags=risk_tags,
        confidence_bucket=conf_bucket,
        risks=["Đây là tín hiệu nghiên cứu, không phải khuyến nghị đầu tư."],
        indicators={"ema20": ema20, "ema50": ema50, "rsi14": rsi14, "atr14": atr14},
        debug_fields=debug_fields,
        latest_price=close,
        marker_time=str(last.get("date") or last.get("timestamp")),
    )
    _persist_signal_audit(market=market, symbol=symbol, timeframe=timeframe, model_id=model_id, signal=out.signal, confidence_bucket=out.confidence_bucket, confidence_score=total_score, reason_short=out.reason_short, risk_tags=out.risk_tags, debug_fields=out.debug_fields, config_hash=config_hash, dataset_h=dataset_h, code_hash=code_hash)
    return out
