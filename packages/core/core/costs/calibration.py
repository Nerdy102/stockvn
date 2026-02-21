from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sqlmodel import Session, select

from core.costs.models import CostCalibrationReport
from core.oms.models import Order
from core.tca.models import OrderTCA


def _winsorize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    q1 = np.percentile(x, lo)
    q2 = np.percentile(x, hi)
    return np.clip(x, q1, q2)


def _load_cfg(path: str = "configs/cost_calibration.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def run_cost_calibration(session: Session, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(cfg or _load_cfg())
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"status": "SKIP", "message": "Hiệu chỉnh đang tắt trong cấu hình."}

    lookback_orders = min(1000, max(1, int(cfg.get("lookback_orders", 200))))
    min_orders = int(cfg.get("min_orders", 30))
    lo, hi = (cfg.get("winsor_pctl") or [1, 99])[:2]
    bounds = cfg.get("param_bounds") or {}

    rows = session.exec(select(OrderTCA).order_by(OrderTCA.created_at.desc()).limit(lookback_orders)).all()
    points: list[tuple[float, float, float]] = []
    for r in rows:
        if r.notional_arrival <= 0:
            continue
        ord_row = session.get(Order, r.order_id)
        if ord_row is None:
            continue
        debug = dict(ord_row.risk_tags_json or {})
        atr_pct = float(debug.get("atr_pct", 0.0))
        order_notional = float(debug.get("order_notional", max(ord_row.qty * (ord_row.price or 0.0), 0.0)))
        dollar_volume_window = float(debug.get("dollar_volume_window", 1.0))
        x1 = atr_pct * 10000.0
        x2 = order_notional / max(dollar_volume_window, 1.0)
        y = max(0.0, (float(r.impact_cost) / max(float(r.notional_arrival), 1e-9)) * 10000.0)
        points.append((y, x1, x2))

    if len(points) < min_orders:
        return {"status": "SKIP", "message": f"Không đủ mẫu để fit (n={len(points)} < {min_orders}).", "n_samples": len(points)}

    arr = np.array(points, dtype=float)
    y = _winsorize(arr[:, 0], lo, hi)
    x1 = _winsorize(arr[:, 1], lo, hi)
    x2 = _winsorize(arr[:, 2], lo, hi)

    X = np.column_stack([np.ones(len(y)), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def clamp(v: float, name: str) -> float:
        b = bounds.get(name) or [v, v]
        return float(min(float(b[1]), max(float(b[0]), float(v))))

    params_new = {
        "base_bps": clamp(float(beta[0]), "base_bps"),
        "k_atr": clamp(float(beta[1]), "k_atr"),
        "k_part": clamp(float(beta[2]), "k_part"),
    }
    params_old = {"base_bps": 3.0, "k_atr": 0.03, "k_part": 50.0}

    y_hat = X @ np.array([params_new["base_bps"], params_new["k_atr"], params_new["k_part"]], dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    medae = float(np.median(np.abs(y - y_hat)))

    rec = "Đề xuất cập nhật tham số để giảm chênh lệch giữa ước lượng và chi phí thực tế trên Paper/Sandbox. Chỉ áp dụng thủ công sau khi kiểm định lại."
    metrics = {
        "r2": r2,
        "median_abs_error": medae,
        "n_samples": len(y),
        "params_old": params_old,
        "recommendation_vi": rec,
    }
    report = CostCalibrationReport(params_new_json=params_new, metrics_json=metrics)
    session.add(report)
    session.commit()
    session.refresh(report)

    if os.getenv("ENABLE_WRITE_OVERRIDES", "false").lower() == "true":
        Path("configs/cost_model_overrides.yaml").write_text(yaml.safe_dump(params_new, allow_unicode=True, sort_keys=True), encoding="utf-8")

    return {"status": "OK", "report_id": report.id, "params_new": params_new, "metrics": metrics}


def load_cost_model_override() -> dict[str, Any] | None:
    if os.getenv("ENABLE_COST_MODEL_OVERRIDE", "false").lower() != "true":
        return None
    p = Path("configs/cost_model_overrides.yaml")
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8")) or None
