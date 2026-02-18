from __future__ import annotations

import datetime as dt
from typing import Any

from core.db.models import DataHealthIncident
from core.monitoring.data_health import compute_incident_sla_gauges
from core.observability.slo import INCIDENT_RULES, load_snapshots
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/data/health", tags=["data_health"])


def _realtime_ops_payload(db: Session) -> dict[str, Any]:
    snapshots = load_snapshots([
        "artifacts/metrics/gateway_metrics.json",
        "artifacts/metrics/bar_builder_metrics.json",
        "artifacts/metrics/signal_engine_metrics.json",
    ])
    rows = db.exec(
        select(DataHealthIncident)
        .where(DataHealthIncident.source.like("realtime_ops:%"))
        .order_by(DataHealthIncident.created_at.desc())
        .limit(100)
    ).all()
    incidents = [
        {
            "id": int(r.id or 0),
            "source": r.source,
            "severity": r.severity,
            "status": r.status,
            "summary": r.summary,
            "runbook_id": r.runbook_section,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]

    gauges = {
        "ingest_lag_s_p95": 0.0,
        "bar_build_latency_s_p95": 0.0,
        "signal_latency_s_p95": 0.0,
        "redis_stream_pending": 0.0,
    }
    if snapshots:
        gauges["ingest_lag_s_p95"] = max(float(s.get("ingest_lag_s_p95", 0.0)) for s in snapshots)
        gauges["bar_build_latency_s_p95"] = max(float(s.get("bar_build_latency_s_p95", 0.0)) for s in snapshots)
        gauges["signal_latency_s_p95"] = max(float(s.get("signal_latency_s_p95", 0.0)) for s in snapshots)
        gauges["redis_stream_pending"] = max(float(s.get("redis_stream_pending", 0.0)) for s in snapshots)

    return {
        "gauges": gauges,
        "incidents": incidents[:20],
        "runbooks": {r.code: r.runbook_id for r in INCIDENT_RULES},
        "snapshots": snapshots,
    }


@router.get("/summary")
def data_health_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.exec(select(DataHealthIncident).order_by(DataHealthIncident.created_at.desc()).limit(500)).all()
    payload = [
        {
            "id": int(r.id or 0),
            "source": r.source,
            "severity": r.severity,
            "status": r.status,
            "symbol": r.symbol,
            "summary": r.summary,
            "runbook_section": r.runbook_section,
            "created_at": r.created_at,
        }
        for r in rows
    ]
    gauges = compute_incident_sla_gauges(payload, now=dt.datetime.utcnow())

    by_source: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for r in payload:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_sev[r["severity"]] = by_sev.get(r["severity"], 0) + 1

    return {
        "as_of": dt.datetime.utcnow().isoformat(),
        "open_incidents": [r for r in payload if str(r["status"]).upper() == "OPEN"][:50],
        "sla_gauges": gauges,
        "counts": {"by_source": by_source, "by_severity": by_sev},
        "realtime_ops": _realtime_ops_payload(db),
    }


@router.get("/incidents")
def data_health_incidents(
    status: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    q = select(DataHealthIncident)
    if status:
        q = q.where(DataHealthIncident.status == status)
    rows = db.exec(q.order_by(DataHealthIncident.created_at.desc()).limit(limit)).all()
    return [
        {
            "id": int(r.id or 0),
            "source": r.source,
            "severity": r.severity,
            "status": r.status,
            "symbol": r.symbol,
            "summary": r.summary,
            "details_json": r.details_json,
            "runbook_section": r.runbook_section,
            "suggested_actions_json": r.suggested_actions_json,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/incidents/{incident_id}")
def data_health_incident_detail(incident_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.exec(select(DataHealthIncident).where(DataHealthIncident.id == incident_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return {
        "id": int(row.id or 0),
        "source": row.source,
        "severity": row.severity,
        "status": row.status,
        "symbol": row.symbol,
        "summary": row.summary,
        "details_json": row.details_json,
        "runbook_section": row.runbook_section,
        "suggested_actions_json": row.suggested_actions_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/realtime_ops")
def data_health_realtime_ops(db: Session = Depends(get_db)) -> dict[str, Any]:
    return _realtime_ops_payload(db)
