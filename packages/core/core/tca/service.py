from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from typing import Any

from sqlmodel import Session, delete, select

from core.db.models import PriceOHLCV
from core.oms.models import Fill, Order, OrderEvent
from core.tca.models import OrderTCA, TCABenchmarkPoint


def _code_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _h(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _quality(is_bps_total: float) -> tuple[str, str]:
    if is_bps_total <= 10:
        return "Tốt", "Khớp gần giá quyết định, chi phí thấp."
    if is_bps_total <= 30:
        return "Vừa", "Có trượt giá/chi phí ở mức trung bình."
    return "Xấu", "Trượt giá/chi phí cao — cân nhắc giảm khối lượng hoặc tăng giới hạn rủi ro."


def _side_sign(side: str) -> int:
    return 1 if side.upper() == "BUY" else -1


def _bar_price_ref(bar: PriceOHLCV | None) -> float:
    if bar is None:
        return 0.0
    return float(bar.close)


def _spread_bps_est(order: Order) -> float:
    # fallback deterministic constant khi chưa có model theo bucket.
    return 5.0 if order.market == "vn" else 8.0


def compute_tca_for_order(session: Session, order_id: str, *, with_points: bool = True) -> OrderTCA | None:
    order = session.get(Order, order_id)
    if order is None:
        return None
    fills = session.exec(select(Fill).where(Fill.order_id == order_id).order_by(Fill.ts.asc())).all()
    if not fills:
        return None

    qty_filled = float(sum(float(f.fill_qty) for f in fills))
    if qty_filled <= 0:
        return None

    events = session.exec(select(OrderEvent).where(OrderEvent.order_id == order_id).order_by(OrderEvent.ts.asc())).all()
    approved = next((e for e in events if e.to_status == "APPROVED"), None)
    arrival_ts = approved.ts if approved else order.updated_at

    exec_start_ts = fills[0].ts
    exec_end_ts = fills[-1].ts

    fill_notional = sum(float(f.fill_qty) * float(f.fill_price) for f in fills)
    exec_vwap_price = fill_notional / max(qty_filled, 1e-9)

    bars = session.exec(
        select(PriceOHLCV)
        .where(PriceOHLCV.symbol == order.symbol)
        .where(PriceOHLCV.timeframe == order.timeframe)
        .order_by(PriceOHLCV.timestamp.asc())
    ).all()
    if not bars:
        return None

    approved_bar = None
    prev_bar = None
    next_bar = None
    for b in bars:
        if b.timestamp <= arrival_ts:
            prev_bar = b
        if approved_bar is None and b.timestamp >= arrival_ts:
            approved_bar = b
        if b.timestamp > arrival_ts:
            next_bar = b
            break

    execution_pref = str(getattr(order, "execution_pref", "close") or "close").lower()
    arrival_price = 0.0
    if float(order.notional_est or 0.0) > 0 and float(order.qty or 0.0) > 0:
        arrival_price = float(order.notional_est) / max(float(order.qty), 1e-9)
    if arrival_price <= 0:
        if execution_pref in {"next_bar", "thanh nến kế tiếp (next-bar)"} and next_bar is not None:
            arrival_price = float(next_bar.open)
        elif approved_bar is not None:
            arrival_price = float(approved_bar.close)
        elif prev_bar is not None:
            arrival_price = float(prev_bar.close)
        else:
            arrival_price = float(bars[0].close)

    in_window = [b for b in bars if exec_start_ts <= b.timestamp <= exec_end_ts]
    if not in_window:
        in_window = [prev_bar] if prev_bar is not None else [bars[-1]]
    refs = [_bar_price_ref(b) for b in in_window]
    benchmark_twap_price = sum(refs) / max(len(refs), 1)

    vols = [float(b.volume or 0.0) for b in in_window]
    typicals = [float((b.high + b.low + b.close) / 3.0) for b in in_window]
    denom = sum(vols)
    if denom > 0:
        benchmark_vwap_price = sum(tp * v for tp, v in zip(typicals, vols)) / denom
    else:
        benchmark_vwap_price = benchmark_twap_price

    fee_total = float(sum(float(f.fee) for f in fills))
    tax_total = float(sum(float(f.tax) for f in fills))
    slippage_total = float(sum(float(f.slippage_cost) for f in fills))
    funding_total = float(sum(float(f.funding_cost) for f in fills))

    sign = _side_sign(order.side)
    is_price_component = sign * (exec_vwap_price - arrival_price) * qty_filled
    explicit_costs = fee_total + tax_total + slippage_total + funding_total
    is_total = is_price_component + explicit_costs
    notional_arrival = arrival_price * qty_filled
    is_bps_total = (is_total / max(notional_arrival, 1e-9)) * 10000.0

    spread_bps_est = _spread_bps_est(order)
    spread_cost = notional_arrival * spread_bps_est / 10000.0
    start_price = _bar_price_ref(in_window[0])
    delay_cost = sign * (start_price - arrival_price) * qty_filled
    impact_cost = is_price_component - delay_cost - spread_cost

    quality_bucket, reason_vi = _quality(is_bps_total)

    cfg_hash = _h({"spread_bps_est": spread_bps_est, "execution": execution_pref})
    data_hash = _h({"fills": [f.model_dump() for f in fills], "bars": [{"ts": b.timestamp.isoformat(), "close": b.close, "volume": b.volume} for b in in_window]})
    code_hash = _code_hash()
    report_id = _h({"order_id": order.id, "cfg": cfg_hash, "data": data_hash, "code": code_hash})

    old = session.exec(select(OrderTCA).where(OrderTCA.order_id == order_id)).first()
    if old is not None:
        session.delete(old)
    if with_points:
        session.exec(delete(TCABenchmarkPoint).where(TCABenchmarkPoint.order_id == order_id))

    row = OrderTCA(
        order_id=order_id,
        market=order.market,
        symbol=order.symbol,
        timeframe=order.timeframe,
        arrival_ts=arrival_ts,
        arrival_price=arrival_price,
        exec_start_ts=exec_start_ts,
        exec_end_ts=exec_end_ts,
        exec_vwap_price=exec_vwap_price,
        benchmark_twap_price=benchmark_twap_price,
        benchmark_vwap_price=benchmark_vwap_price,
        side=order.side,
        qty_requested=float(order.qty),
        qty_filled=qty_filled,
        notional_arrival=notional_arrival,
        fee_total=fee_total,
        tax_total=tax_total,
        slippage_total=slippage_total,
        funding_total=funding_total,
        is_price_component=is_price_component,
        is_total=is_total,
        is_bps_total=is_bps_total,
        spread_bps_est=spread_bps_est,
        delay_cost=delay_cost,
        impact_cost=impact_cost,
        quality_bucket=quality_bucket,
        reason_vi=reason_vi,
        config_hash=cfg_hash,
        dataset_hash=data_hash,
        code_hash=code_hash,
        report_id=report_id,
    )
    session.add(row)

    if with_points:
        for b, tp in zip(in_window, typicals):
            session.add(
                TCABenchmarkPoint(
                    order_id=order_id,
                    ts=b.timestamp,
                    bar_price_ref=float(b.close),
                    bar_volume=float(b.volume or 0.0),
                    bar_vwap_est=float(tp),
                )
            )
    session.commit()
    session.refresh(row)
    return row
