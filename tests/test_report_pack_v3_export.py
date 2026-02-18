from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from core.db.models import (
    BacktestEquityCurve,
    BacktestMetric,
    BacktestRun,
    DataHealthIncident,
    DiagnosticsMetric,
    DiagnosticsRun,
    GateResult,
)
from core.db.session import create_db_and_tables, get_engine
from core.report_pack_v3 import SECTION_ORDER, export_report_pack_v3
from sqlmodel import Session


def test_report_pack_v3_contains_required_sections_and_manifest_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "report_v3.db"
    exec_model_path = tmp_path / "execution_model.yaml"
    exec_model_path.write_text("base_slippage_bps: 10\n", encoding="utf-8")

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("EXECUTION_MODEL_PATH", str(exec_model_path))

    create_db_and_tables(f"sqlite:///{db_path}")
    engine = get_engine(f"sqlite:///{db_path}")

    run_hash = "run-v3-hash"
    with Session(engine) as session:
        bt = BacktestRun(
            run_hash=run_hash,
            config_json={"rebalance": "monthly", "cost_bps": 12},
            summary_json={"cagr": 0.15},
        )
        session.add(bt)
        session.commit()
        session.refresh(bt)

        session.add(
            BacktestMetric(run_id=int(bt.id), metric_name="sharpe", metric_value=1.23)
        )
        session.add(
            BacktestEquityCurve(
                run_id=int(bt.id),
                date=dt.date(2025, 1, 2),
                equity=101_000_000,
            )
        )
        session.add(
            DiagnosticsRun(
                run_id=run_hash,
                model_id="alpha_v3",
                config_hash="cfg123",
            )
        )
        session.add(
            DiagnosticsMetric(
                run_id=run_hash,
                metric_name="risk_vol_20d",
                metric_value=0.22,
            )
        )
        session.add(
            DiagnosticsMetric(
                run_id=run_hash,
                metric_name="cost_turnover_bps",
                metric_value=18.0,
            )
        )
        session.add(
            DiagnosticsMetric(
                run_id=run_hash,
                metric_name="ece_10",
                metric_value=0.03,
            )
        )
        session.add(
            GateResult(
                run_id=run_hash,
                status="PASS",
                reasons={"pbo": "ok"},
                details={"threshold": 0.2},
            )
        )
        session.add(
            DataHealthIncident(
                source="unit_test",
                severity="HIGH",
                status="OPEN",
                summary="stale feed",
                details_json={"lag_min": 20},
                runbook_section="DH-200",
            )
        )
        session.commit()

    bundle = export_report_pack_v3(run_hash, outdir=tmp_path / "bundle")

    html = bundle.html_path.read_text(encoding="utf-8")
    for section in SECTION_ORDER:
        assert f"<h2>{section.title()}</h2>" in html

    assert bundle.pdf_path.read_bytes().startswith(b"%PDF")

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["report_pack_version"] == "v3"
    assert manifest["sections"] == SECTION_ORDER
    assert manifest["hashes"]["files"]["report_v3.html"]
    assert manifest["hashes"]["files"]["report_v3.pdf"]
    assert "model_versions_hash" in manifest["hashes"]
