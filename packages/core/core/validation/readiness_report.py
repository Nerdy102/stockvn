from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlmodel import Session, select

from core.monitoring.drift_monitor import DriftAlertTrade
from data.quality.models import DataQualityEvent


def _hash_obj(v: object) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_readiness_report(
    *,
    walk_forward: dict[str, Any],
    stress: dict[str, Any],
    db: Session,
    model_id: str,
    market: str,
) -> dict[str, Any]:
    dq = db.exec(select(DataQualityEvent)).all()
    by_sev: dict[str, int] = {}
    for e in dq:
        by_sev[e.severity] = by_sev.get(e.severity, 0) + 1

    alerts = db.exec(
        select(DriftAlertTrade).where(DriftAlertTrade.model_id == model_id, DriftAlertTrade.market == market)
    ).all()
    drift = {
        "high_alerts": sum(1 for a in alerts if a.severity == "HIGH"),
        "latest_codes": [a.code for a in alerts[-3:]],
    }

    obj = {
        "walk_forward": {
            "stability_score": walk_forward.get("stability_score", 0.0),
            "per_fold_summary": walk_forward.get("per_fold_metrics", []),
        },
        "stress": {
            "worst_case_metrics": stress.get("worst_case", {}),
            "sensitivity_index": stress.get("sensitivity_index", 0.0),
        },
        "data_quality": {"count_events_by_severity": by_sev},
        "paper_vs_backtest_drift": drift,
    }
    payload_hash = _hash_obj(obj)
    obj["hashes"] = {"payload_hash": payload_hash}
    obj["report_id"] = _hash_obj({"payload_hash": payload_hash, "model_id": model_id, "market": market})
    return obj
