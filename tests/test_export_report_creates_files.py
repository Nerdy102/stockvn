from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.db.models import DiagnosticsRun
from core.db.session import create_db_and_tables, get_engine
from sqlmodel import Session


def test_export_alpha_report_creates_bundle(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{db_path}"

    create_db_and_tables(env["DATABASE_URL"])
    engine = get_engine(env["DATABASE_URL"])
    with Session(engine) as s:
        s.add(DiagnosticsRun(run_id="demo-run", model_id="ensemble_v2", config_hash="x"))
        s.commit()

    proc = subprocess.run(
        [sys.executable, "scripts/export_alpha_report.py", "--run-id", "demo-run"],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    outdir = Path(__file__).resolve().parents[1] / "artifacts" / "reports" / "demo-run"
    assert (outdir / "report.html").exists()
    assert (outdir / "metrics_table.csv").exists()
    assert (outdir / "equity_curve.csv").exists()
    assert (outdir / "diagnostics.csv").exists()
    assert (outdir / "dsr_pbo_gates.csv").exists()
