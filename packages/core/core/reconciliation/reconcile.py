from __future__ import annotations

import datetime as dt
from typing import Any

from sqlmodel import Session, select

from core.db.models import (
    DataHealthIncident,
    GovernanceState,
    PriceOHLCV,
    ReconciliationRun,
    Trade,
)
from core.fees_taxes import FeesTaxes


def _latest_marks(session: Session) -> dict[str, float]:
    rows = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    out: dict[str, tuple[dt.datetime, float]] = {}
    for r in rows:
        cur = out.get(r.symbol)
        if cur is None or r.timestamp > cur[0]:
            out[r.symbol] = (r.timestamp, float(r.close))
    return {k: v for k, (_, v) in out.items()}


def reconcile_portfolio(
    session: Session,
    *,
    portfolio_id: int,
    broker_name: str,
    fees: FeesTaxes,
    expected_equity: float | None = None,
    tolerance_vnd: float = 1.0,
) -> dict[str, Any]:
    trades = session.exec(select(Trade).where(Trade.portfolio_id == portfolio_id)).all()
    marks = _latest_marks(session)

    cash = 1_000_000_000.0
    qty: dict[str, float] = {}
    for t in sorted(trades, key=lambda x: (x.trade_date, x.id or 0)):
        side = str(t.side).upper()
        q = float(t.quantity)
        px = float(t.price)
        notional = q * px
        comm = fees.commission(notional, broker_name)
        tax = fees.sell_tax(notional) if side == "SELL" else 0.0
        if side == "BUY":
            qty[t.symbol] = qty.get(t.symbol, 0.0) + q
            cash -= notional + comm + tax
        else:
            current = qty.get(t.symbol, 0.0)
            exec_q = min(current, q)  # SELL clamp bugfix
            qty[t.symbol] = current - exec_q
            cash += exec_q * px - comm - tax

    negatives = {k: v for k, v in qty.items() if v < -1e-9}
    mtm = sum(max(v, 0.0) * float(marks.get(k, 0.0)) for k, v in qty.items())
    equity = cash + mtm
    target = equity if expected_equity is None else float(expected_equity)
    gap = float(equity - target)
    mismatch = abs(gap) > float(tolerance_vnd) or bool(negatives)

    run = ReconciliationRun(
        portfolio_id=portfolio_id,
        status="MISMATCH" if mismatch else "OK",
        tolerance_vnd=float(tolerance_vnd),
        equity_gap=gap,
        details_json={"cash": cash, "mtm": mtm, "equity": equity, "qty": qty, "negatives": negatives},
    )
    session.add(run)

    incident_id = None
    if mismatch:
        inc = DataHealthIncident(
            source="reconciliation",
            severity="HIGH",
            status="OPEN",
            summary=f"Reconciliation mismatch portfolio={portfolio_id}",
            details_json={"equity": equity, "target": target, "gap": gap, "negatives": negatives},
            runbook_section="OPS-RECON-001",
            suggested_actions_json={"actions": ["Pause OMS", "Check fills/orders", "Reconcile cash and positions"]},
        )
        session.add(inc)
        gov = session.exec(select(GovernanceState).order_by(GovernanceState.updated_at.desc())).first()
        if gov is None:
            gov = GovernanceState(status="PAUSED", pause_reason="reconciliation_mismatch", source="reconciliation")
            session.add(gov)
        else:
            gov.status = "PAUSED"
            gov.pause_reason = "reconciliation_mismatch"
            gov.source = "reconciliation"
            gov.updated_at = dt.datetime.utcnow()
        session.flush()
        incident_id = inc.id

    session.commit()
    session.refresh(run)
    return {
        "run_id": run.id,
        "status": run.status,
        "equity": equity,
        "target": target,
        "gap": gap,
        "incident_id": incident_id,
        "governance_paused": mismatch,
    }
