from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from core.db.models import (
    BacktestEquityCurve,
    BacktestMetric,
    BacktestRun,
    DiagnosticsMetric,
    DiagnosticsRun,
    DsrResult,
    GateResult,
    MinTrlResult,
    PboResult,
    PsrResult,
    RealityCheckResult,
    SpaResult,
)
from core.db.session import create_db_and_tables, get_engine
from core.settings import get_settings
from sqlmodel import Session, select


def _as_df(rows: list[object]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in rows]) if rows else pd.DataFrame()


def _resolve_backtest_run_id(session: Session, run_id: str) -> int | None:
    if run_id.isdigit():
        return int(run_id)

    # Prefer explicit mapping via run_hash if available.
    run = session.exec(select(BacktestRun).where(BacktestRun.run_hash == run_id)).first()
    if run is not None and run.id is not None:
        return int(run.id)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export alpha report bundle (HTML+CSV)")
    parser.add_argument("--run-id", required=True, help="Diagnostics run_id or numeric backtest run id")
    args = parser.parse_args()

    settings = get_settings()
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)

    outdir = Path("artifacts/reports") / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        diag_run = session.exec(
            select(DiagnosticsRun).where(DiagnosticsRun.run_id == args.run_id)
        ).first()

        diagnostics = session.exec(
            select(DiagnosticsMetric).where(DiagnosticsMetric.run_id == args.run_id)
        ).all()
        dsr = session.exec(select(DsrResult).where(DsrResult.run_id == args.run_id)).first()
        pbo = session.exec(select(PboResult).where(PboResult.run_id == args.run_id)).first()
        psr = session.exec(select(PsrResult).where(PsrResult.run_id == args.run_id)).first()
        mintrl = session.exec(select(MinTrlResult).where(MinTrlResult.run_id == args.run_id)).first()
        rc = session.exec(select(RealityCheckResult).where(RealityCheckResult.run_id == args.run_id)).first()
        spa = session.exec(select(SpaResult).where(SpaResult.run_id == args.run_id)).first()
        gate = session.exec(select(GateResult).where(GateResult.run_id == args.run_id)).first()

        bt_run_id = _resolve_backtest_run_id(session, args.run_id)
        metrics_rows: list[BacktestMetric] = []
        eq_rows: list[BacktestEquityCurve] = []
        if bt_run_id is not None:
            metrics_rows = session.exec(
                select(BacktestMetric).where(BacktestMetric.run_id == bt_run_id)
            ).all()
            eq_rows = session.exec(
                select(BacktestEquityCurve).where(BacktestEquityCurve.run_id == bt_run_id)
            ).all()

        if diag_run is None and bt_run_id is None:
            raise SystemExit(f"run_id not found: {args.run_id}")

    metrics_df = _as_df(metrics_rows)
    eq_df = _as_df(eq_rows)
    diag_df = _as_df(diagnostics)

    gates = {
        "run_id": args.run_id,
        "backtest_run_id": bt_run_id,
        "dsr": float(dsr.dsr_value) if dsr else None,
        "pbo": float(pbo.phi) if pbo else None,
        "psr": float(psr.psr_value) if psr else None,
        "mintrl": float(mintrl.mintrl) if mintrl else None,
        "rc_p": float(rc.p_value) if rc else None,
        "spa_p": float(spa.p_value) if spa else None,
        "gate_status": None if gate is None else gate.status,
        "gate_reasons": None if gate is None else gate.reasons,
    }
    gates_df = pd.DataFrame([gates])

    metrics_df.to_csv(outdir / "metrics_table.csv", index=False)
    eq_df.to_csv(outdir / "equity_curve.csv", index=False)
    diag_df.to_csv(outdir / "diagnostics.csv", index=False)
    gates_df.to_csv(outdir / "dsr_pbo_gates.csv", index=False)

    html = outdir / "report.html"
    html.write_text(
        "<h1>Alpha Report Export</h1>"
        + "<h2>Metrics table</h2>"
        + metrics_df.to_html(index=False)
        + "<h2>Equity curve</h2>"
        + eq_df.to_html(index=False)
        + "<h2>Diagnostics</h2>"
        + diag_df.to_html(index=False)
        + "<h2>DSR / PBO / Gates</h2>"
        + gates_df.to_html(index=False),
        encoding="utf-8",
    )

    print(f"Exported bundle in {outdir}")


if __name__ == "__main__":
    main()
