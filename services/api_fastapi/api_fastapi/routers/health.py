from __future__ import annotations

import datetime as dt
import time
from typing import Any

from core.oms.models import Order
from core.reconciliation.models import ReconcileReport
from core.risk.controls_models import TradingControl
from core.settings import get_settings
from data.providers.factory import get_provider
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlmodel import Session, SQLModel, select

from api_fastapi.deps import get_db

router = APIRouter(tags=["health"])


def _control_state(db: Session) -> TradingControl:
    try:
        SQLModel.metadata.create_all(db.get_bind(), tables=[TradingControl.__table__, ReconcileReport.__table__, Order.__table__])
    except Exception:
        pass
    row = db.get(TradingControl, 1)
    if row is None:
        row = TradingControl(id=1, kill_switch_enabled=False, paused_reason_code=None)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def _freshness() -> tuple[bool, str | None]:
    settings = get_settings()
    provider = get_provider(settings)
    sym = "VNINDEX" if settings.TRADING_ENV != "live" else "BTCUSDT"
    tf = "1D" if sym == "VNINDEX" else "60m"
    df = provider.get_ohlcv(sym, tf)
    if df is None or df.empty:
        return False, None
    stamp = str(df.iloc[-1].get("date") or df.iloc[-1].get("timestamp"))
    if tf == "1D":
        try:
            as_of = dt.date.fromisoformat(stamp[:10])
            return (dt.date.today() - as_of).days <= 2, stamp
        except Exception:
            return False, stamp
    try:
        as_of_ts = dt.datetime.fromisoformat(stamp)
        return (dt.datetime.utcnow() - as_of_ts).total_seconds() <= 6 * 3600, stamp
    except Exception:
        return False, stamp


@router.get("/health")
def health_legacy(db: Session = Depends(get_db)) -> dict[str, Any]:
    return healthz(db)


@router.get("/healthz")
def healthz(db: Session = Depends(get_db)) -> dict[str, Any]:
    t0 = time.perf_counter()
    db_ok = True
    try:
        db.exec(text("SELECT 1"))
    except Exception:
        db_ok = False
    db_latency_ms = (time.perf_counter() - t0) * 1000.0
    control = _control_state(db)
    freshness_ok, as_of_value = _freshness()

    fail = (not db_ok) or (not freshness_ok)
    return {
        "status": "FAIL" if fail else "OK",
        "db_ok": db_ok,
        "db_latency_ms": round(db_latency_ms, 3),
        "data_freshness_ok": freshness_ok,
        "as_of": as_of_value,
        "kill_switch_state": bool(control.kill_switch_enabled),
    }


@router.get("/healthz/detail")
def healthz_detail(db: Session = Depends(get_db)) -> dict[str, Any]:
    settings = get_settings()
    t0 = time.perf_counter()
    db_ok = True
    try:
        db.exec(text("SELECT 1"))
    except Exception:
        db_ok = False
    db_latency_ms = (time.perf_counter() - t0) * 1000.0

    control = _control_state(db)
    last_reconcile = db.exec(select(ReconcileReport).order_by(ReconcileReport.ts.desc())).first()
    freshness_ok, as_of_value = _freshness()
    broker_ok = True if settings.TRADING_ENV in {"paper", "dev"} or settings.ENABLE_SANDBOX else False

    return {
        "db_ok": db_ok,
        "db_latency_ms": round(db_latency_ms, 3),
        "broker_ok": broker_ok,
        "data_freshness_ok": freshness_ok,
        "as_of_date": as_of_value,
        "kill_switch_state": bool(control.kill_switch_enabled),
        "drift_pause_state": bool((control.paused_reason_code or "").upper() == "DRIFT_PAUSED"),
        "last_reconcile_ts": None if last_reconcile is None else last_reconcile.ts.isoformat(),
    }


@router.post("/reconcile/run")
def run_reconcile(db: Session = Depends(get_db)) -> dict[str, Any]:
    from jobs.reconcile import run_reconciliation

    report = run_reconciliation(db)
    return {
        "message": "Đã chạy đối soát (reconciliation).",
        "status": report.status,
        "ts": report.ts.isoformat(),
        "mismatches": report.mismatches_json,
    }


@router.get("/reconcile/latest")
def latest_reconcile(db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.exec(select(ReconcileReport).order_by(ReconcileReport.ts.desc())).first()
    if row is None:
        return {"status": "NO_DATA", "mismatch_count": 0, "last_reconcile_ts": None}
    mismatches = row.mismatches_json or {}
    mismatch_count = sum(len(v) for v in mismatches.values() if isinstance(v, list))
    return {
        "status": row.status,
        "mismatch_count": mismatch_count,
        "last_reconcile_ts": row.ts.isoformat(),
        "mismatches": mismatches,
    }
