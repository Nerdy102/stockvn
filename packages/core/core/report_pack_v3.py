from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.db.models import (
    AlphaModel,
    BacktestEquityCurve,
    BacktestMetric,
    BacktestRun,
    DataHealthIncident,
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

SECTION_ORDER = [
    "summary",
    "equity",
    "costs",
    "risk",
    "gates",
    "calibration",
    "health",
    "assumptions",
]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _read_hash(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return _sha256_bytes(p.read_bytes())


def _escape_html(text: Any) -> str:
    s = str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{_escape_html(c)}</th>" for c in cols)
    body_rows = []
    for r in rows:
        tds = "".join(f"<td>{_escape_html(r.get(c, ''))}</td>" for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_text_pdf(lines: list[str]) -> bytes:
    safe_lines = [ln if ln else " " for ln in lines]
    y = 790
    content_parts = ["BT", "/F1 11 Tf"]
    for ln in safe_lines:
        if y < 40:
            break
        content_parts.append(f"1 0 0 1 40 {y} Tm ({_pdf_escape(ln[:120])}) Tj")
        y -= 14
    content_parts.append("ET")
    stream_data = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream_data)} >> stream\n".encode("ascii") + stream_data + b"\nendstream endobj\n",
    ]

    out = bytearray(b"%PDF-1.4\n")
    xref = [0]
    for obj in objects:
        xref.append(len(out))
        out.extend(obj)
    xref_start = len(out)
    out.extend(f"xref\n0 {len(xref)}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in xref[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer << /Size {len(xref)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


def _resolve_backtest_run_id(session: Session, run_id: str) -> int | None:
    if run_id.isdigit():
        return int(run_id)
    run = session.exec(select(BacktestRun).where(BacktestRun.run_hash == run_id)).first()
    if run is not None and run.id is not None:
        return int(run.id)
    return None


def _as_rows(items: list[Any]) -> list[dict[str, Any]]:
    return [x.model_dump() for x in items]


@dataclass
class ReportPackV3:
    outdir: Path
    html_path: Path
    pdf_path: Path
    manifest_path: Path


def export_report_pack_v3(run_id: str, outdir: Path | None = None) -> ReportPackV3:
    settings = get_settings()
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)

    target_dir = outdir or (Path("artifacts/reports_v3") / run_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        diag_run = session.exec(select(DiagnosticsRun).where(DiagnosticsRun.run_id == run_id)).first()
        bt_run_id = _resolve_backtest_run_id(session, run_id)

        if diag_run is None and bt_run_id is None:
            raise ValueError(f"run_id not found: {run_id}")

        metrics_rows = []
        eq_rows = []
        bt_config = {}
        bt_summary = {}

        if bt_run_id is not None:
            metrics_rows = _as_rows(session.exec(select(BacktestMetric).where(BacktestMetric.run_id == bt_run_id)).all())
            eq_rows = _as_rows(session.exec(select(BacktestEquityCurve).where(BacktestEquityCurve.run_id == bt_run_id)).all())
            bt_run = session.exec(select(BacktestRun).where(BacktestRun.id == bt_run_id)).first()
            if bt_run is not None:
                bt_config = bt_run.config_json or {}
                bt_summary = bt_run.summary_json or {}

        diagnostics = _as_rows(session.exec(select(DiagnosticsMetric).where(DiagnosticsMetric.run_id == run_id)).all())
        dsr = session.exec(select(DsrResult).where(DsrResult.run_id == run_id)).first()
        pbo = session.exec(select(PboResult).where(PboResult.run_id == run_id)).first()
        psr = session.exec(select(PsrResult).where(PsrResult.run_id == run_id)).first()
        mintrl = session.exec(select(MinTrlResult).where(MinTrlResult.run_id == run_id)).first()
        rc = session.exec(select(RealityCheckResult).where(RealityCheckResult.run_id == run_id)).first()
        spa = session.exec(select(SpaResult).where(SpaResult.run_id == run_id)).first()
        gate = session.exec(select(GateResult).where(GateResult.run_id == run_id)).first()
        incidents = _as_rows(
            session.exec(
                select(DataHealthIncident).order_by(DataHealthIncident.created_at.desc()).limit(20)
            ).all()
        )
        models = _as_rows(session.exec(select(AlphaModel).order_by(AlphaModel.created_at.desc()).limit(20)).all())

    assumptions = {
        "execution_model_path": settings.EXECUTION_MODEL_PATH,
        "execution_model_hash": _read_hash(settings.EXECUTION_MODEL_PATH),
        "diagnostics_config_hash": diag_run.config_hash if diag_run is not None else None,
        "backtest_config": bt_config,
        "backtest_config_hash": _sha256_json(bt_config),
        "model_versions": [
            {
                "model_id": m.get("model_id"),
                "version": m.get("version"),
                "config_hash": m.get("config_hash"),
            }
            for m in models
        ],
    }

    summary_rows = [
        {
            "run_id": run_id,
            "backtest_run_id": bt_run_id,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "metrics_count": len(metrics_rows),
            "equity_points": len(eq_rows),
            "diagnostics_metrics": len(diagnostics),
            "open_health_incidents": sum(1 for r in incidents if str(r.get("status", "")).upper() == "OPEN"),
        }
    ]

    costs_rows = [r for r in diagnostics if "cost" in str(r.get("metric_name", "")).lower()]
    risk_rows = [r for r in diagnostics if "risk" in str(r.get("metric_name", "")).lower()]
    calibration_rows = [
        r
        for r in diagnostics
        if any(k in str(r.get("metric_name", "")).lower() for k in ("calibration", "ece", "coverage", "interval"))
    ]

    gates_rows = [
        {
            "dsr": None if dsr is None else dsr.dsr_value,
            "pbo": None if pbo is None else pbo.phi,
            "psr": None if psr is None else psr.psr_value,
            "mintrl": None if mintrl is None else mintrl.mintrl,
            "rc_p": None if rc is None else rc.p_value,
            "spa_p": None if spa is None else spa.p_value,
            "gate_status": None if gate is None else gate.status,
            "gate_reasons": None if gate is None else gate.reasons,
            "gate_details": None if gate is None else gate.details,
        }
    ]

    sections: dict[str, list[dict[str, Any]]] = {
        "summary": summary_rows,
        "equity": eq_rows,
        "costs": costs_rows,
        "risk": risk_rows,
        "gates": gates_rows,
        "calibration": calibration_rows,
        "health": incidents,
        "assumptions": [assumptions],
    }

    html_body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Report Pack v3</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} h1{margin-bottom:4px;} h2{margin-top:26px;border-bottom:1px solid #ddd;padding-bottom:4px;} table{border-collapse:collapse;width:100%;font-size:12px;} th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top;} th{background:#f5f5f5;} code{background:#f2f2f2;padding:2px 4px;}</style>",
        "</head><body>",
        "<h1>Report Pack v3</h1><p>Audit-grade export bundle for offline review.</p>",
    ]
    for sec in SECTION_ORDER:
        html_body.append(f"<h2>{sec.title()}</h2>")
        html_body.append(_render_table(sections[sec]))
    html_body.append("</body></html>")
    html_text = "".join(html_body)

    html_path = target_dir / "report_v3.html"
    pdf_path = target_dir / "report_v3.pdf"
    manifest_path = target_dir / "manifest_v3.json"

    html_path.write_text(html_text, encoding="utf-8")

    pdf_lines = ["Report Pack v3", f"run_id: {run_id}", ""]
    for sec in SECTION_ORDER:
        pdf_lines.append(sec.upper())
        rows = sections[sec]
        if not rows:
            pdf_lines.append("- No data")
            continue
        for row in rows[:8]:
            pdf_lines.append("- " + json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)[:110])
        pdf_lines.append("")
    pdf_path.write_bytes(_build_text_pdf(pdf_lines))

    bundle_files = {
        "report_v3.html": _sha256_bytes(html_path.read_bytes()),
        "report_v3.pdf": _sha256_bytes(pdf_path.read_bytes()),
    }

    manifest = {
        "report_pack_version": "v3",
        "run_id": run_id,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sections": SECTION_ORDER,
        "hashes": {
            "files": bundle_files,
            "configs": {
                "diagnostics_config_hash": assumptions["diagnostics_config_hash"],
                "backtest_config_hash": assumptions["backtest_config_hash"],
                "execution_model_hash": assumptions["execution_model_hash"],
            },
            "model_versions_hash": _sha256_json(assumptions["model_versions"]),
        },
        "artifacts": {
            "html": html_path.name,
            "pdf": pdf_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return ReportPackV3(outdir=target_dir, html_path=html_path, pdf_path=pdf_path, manifest_path=manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report pack v3 (HTML+PDF+manifest)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else None
    bundle = export_report_pack_v3(run_id=args.run_id, outdir=outdir)
    print(f"Exported v3 bundle in {bundle.outdir}")


if __name__ == "__main__":
    main()
