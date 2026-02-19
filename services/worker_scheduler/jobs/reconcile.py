from __future__ import annotations

import datetime as dt
from typing import Any

from core.oms.models import Fill, Order, PortfolioSnapshot
from core.reconciliation.models import ReconcileReport
from sqlmodel import Session, SQLModel, select


def run_reconciliation(session: Session) -> ReconcileReport:
    try:
        SQLModel.metadata.create_all(
            session.get_bind(),
            tables=[Order.__table__, Fill.__table__, PortfolioSnapshot.__table__, ReconcileReport.__table__],
        )
    except Exception:
        pass

    mismatches: dict[str, Any] = {"orders_without_fill": [], "snapshot_nav_mismatch": []}
    fixed_actions: dict[str, Any] = {"notes": []}

    orders = session.exec(select(Order)).all()
    for o in orders:
        if o.status in {"FILLED", "PARTIAL_FILLED"}:
            fills = session.exec(select(Fill).where(Fill.order_id == o.id)).all()
            if not fills:
                mismatches["orders_without_fill"].append({"order_id": o.id, "status": o.status})

    snaps = session.exec(select(PortfolioSnapshot)).all()
    for s in snaps:
        positions = s.positions_json or {}
        est_sum = float(s.cash) + sum(float(v) for v in positions.values())
        if abs(est_sum - float(s.nav_est)) > 1e-6:
            mismatches["snapshot_nav_mismatch"].append(
                {"ts": s.ts.isoformat(), "nav_est": s.nav_est, "sum_cash_positions": est_sum}
            )

    status = "OK" if not mismatches["orders_without_fill"] and not mismatches["snapshot_nav_mismatch"] else "MISMATCH"
    report = ReconcileReport(
        ts=dt.datetime.utcnow(),
        status=status,
        mismatches_json=mismatches,
        fixed_actions_json=fixed_actions,
    )
    session.add(report)
    session.commit()
    session.refresh(report)
    return report
