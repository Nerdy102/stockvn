from __future__ import annotations

import datetime as dt
import hashlib
import json
import mimetypes
import subprocess
import uuid
from pathlib import Path
from typing import Any

import yaml
from core.db.models import InteractiveRun
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/lab", tags=["lab_runs"])

ARTIFACT_ROOT = Path("artifacts/runs")
RUN_TYPES = {"RAOCMOE_BACKTEST", "EVAL_LAB", "DATA_INGEST", "SEED_DB"}
CONFIG_MAP = {
    "RAOCMOE_BACKTEST": Path("configs/raocmoe.yaml"),
    "EVAL_LAB": Path("configs/eval_lab.yaml"),
    "DATA_INGEST": Path("configs/providers/data_drop_default.yaml"),
    "SEED_DB": Path("configs/execution_model.yaml"),
}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _safe_run_dir(row: InteractiveRun) -> Path:
    p = Path(row.artifacts_dir).resolve()
    root = ARTIFACT_ROOT.resolve()
    if root not in p.parents and p != root:
        raise HTTPException(status_code=400, detail="invalid artifacts dir")
    return p


def create_interactive_run(db: Session, run_type: str, params_json: dict[str, Any], tags: str | None = None) -> InteractiveRun:
    if run_type not in RUN_TYPES:
        raise HTTPException(status_code=400, detail="unsupported run_type")
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    outdir = ARTIFACT_ROOT / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "params.json").write_text(json.dumps(params_json, sort_keys=True), encoding="utf-8")
    code_hash = _git_head()
    cfg_text = CONFIG_MAP[run_type].read_text(encoding="utf-8") if CONFIG_MAP[run_type].exists() else ""
    row = InteractiveRun(
        run_id=run_id,
        run_type=run_type,
        status="PENDING",
        params_json=params_json,
        config_hash=_sha_text(cfg_text),
        code_hash=code_hash,
        artifacts_dir=str(outdir),
        tags=tags,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.post("/runs")
def create_run(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, str]:
    run_type = str(payload.get("run_type", ""))
    params_json = payload.get("params_json") or {}
    tags = payload.get("tags")
    row = create_interactive_run(db, run_type, params_json, tags)
    return {"run_id": row.run_id, "status": row.status}


@router.get("/runs")
def list_runs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    run_type: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    q = select(InteractiveRun).order_by(InteractiveRun.created_at.desc())
    if run_type:
        q = q.where(InteractiveRun.run_type == run_type)
    if status:
        q = q.where(InteractiveRun.status == status)
    rows = db.exec(q.offset(offset).limit(limit)).all()
    return [r.model_dump() for r in rows]


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.exec(select(InteractiveRun).where(InteractiveRun.run_id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return row.model_dump()


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    row = db.exec(select(InteractiveRun).where(InteractiveRun.run_id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    if row.status == "PENDING":
        row.status = "CANCELLED"
    else:
        params = dict(row.params_json or {})
        params["cancel_requested"] = True
        row.params_json = params
    db.add(row)
    db.commit()
    return {"run_id": run_id, "status": row.status}


@router.get("/runs/{run_id}/log")
def get_run_log(run_id: str, tail: int = 4000, db: Session = Depends(get_db)) -> dict[str, str]:
    row = db.exec(select(InteractiveRun).where(InteractiveRun.run_id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    log_path = _safe_run_dir(row) / "run.log"
    if not log_path.exists():
        return {"log": ""}
    data = log_path.read_text(encoding="utf-8", errors="ignore")
    return {"log": data[-tail:]}


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    row = db.exec(select(InteractiveRun).where(InteractiveRun.run_id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    out = _safe_run_dir(row)
    items = []
    for p in out.rglob("*"):
        if p.is_file():
            rel = p.relative_to(out).as_posix()
            items.append({"path": rel, "size": p.stat().st_size, "mime": mimetypes.guess_type(str(p))[0] or "application/octet-stream"})
    return items


@router.get("/runs/{run_id}/artifact")
def get_artifact(run_id: str, path: str, db: Session = Depends(get_db)):
    row = db.exec(select(InteractiveRun).where(InteractiveRun.run_id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    base = _safe_run_dir(row)
    target = (base / path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(target)


raocmoe_router = APIRouter(prefix="/raocmoe", tags=["raocmoe"])
eval_router = APIRouter(prefix="/eval_lab", tags=["eval_lab"])
config_router = APIRouter(tags=["configs"])
data_router = APIRouter(prefix="/data", tags=["data_manager"])


@raocmoe_router.post("/run")
def raocmoe_run(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, str]:
    row = create_interactive_run(db, "RAOCMOE_BACKTEST", payload)
    return {"run_id": row.run_id, "status": row.status}


@raocmoe_router.get("/runs")
def raocmoe_runs(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    q = select(InteractiveRun).where(InteractiveRun.run_type == "RAOCMOE_BACKTEST").order_by(InteractiveRun.created_at.desc())
    return [r.model_dump() for r in db.exec(q.offset(offset).limit(limit)).all()]


@eval_router.post("/run")
def eval_run(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, str]:
    row = create_interactive_run(db, "EVAL_LAB", payload)
    return {"run_id": row.run_id, "status": row.status}


@config_router.get("/configs")
def get_config(name: str = Query(..., pattern="^(raocmoe|eval_lab|execution|fees_taxes)$")) -> dict[str, str]:
    file_map = {
        "raocmoe": Path("configs/raocmoe.yaml"),
        "eval_lab": Path("configs/eval_lab.yaml"),
        "execution": Path("configs/execution_model.yaml"),
        "fees_taxes": Path("configs/fees_taxes.yaml"),
    }
    p = file_map[name]
    return {"name": name, "yaml": p.read_text(encoding="utf-8")}


@config_router.post("/configs")
def save_config(payload: dict[str, Any], name: str = Query(..., pattern="^(raocmoe|eval_lab|execution|fees_taxes)$")) -> dict[str, str]:
    yaml_text = str(payload.get("yaml", ""))
    yaml.safe_load(yaml_text)
    out = Path("configs/local_overrides")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}.yaml"
    dest.write_text(yaml_text, encoding="utf-8")
    return {"status": "saved", "path": str(dest)}


@config_router.get("/configs/active")
def active_config(name: str = Query(..., pattern="^(raocmoe|eval_lab|execution|fees_taxes)$")) -> dict[str, Any]:
    file_map = {
        "raocmoe": Path("configs/raocmoe.yaml"),
        "eval_lab": Path("configs/eval_lab.yaml"),
        "execution": Path("configs/execution_model.yaml"),
        "fees_taxes": Path("configs/fees_taxes.yaml"),
    }
    base = file_map[name]
    override = Path("configs/local_overrides") / f"{name}.yaml"
    base_obj = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    over_obj = yaml.safe_load(override.read_text(encoding="utf-8")) if override.exists() else {}
    merged = dict(base_obj)
    merged.update(over_obj or {})
    return {"name": name, "override_active": override.exists(), "merged": merged}


@data_router.post("/upload")
async def upload_data(file: UploadFile = File(...)) -> dict[str, str]:
    out = Path("data_drop/inbox")
    out.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.csv").name
    dest = out / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return {"token": safe_name, "path": str(dest)}


@data_router.post("/ingest")
def data_ingest(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, str]:
    row = create_interactive_run(db, "DATA_INGEST", payload)
    return {"run_id": row.run_id, "status": row.status}


@data_router.post("/seed")
def data_seed(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, str]:
    row = create_interactive_run(db, "SEED_DB", payload)
    return {"run_id": row.run_id, "status": row.status}


@data_router.get("/audit/latest")
def data_audit_latest() -> dict[str, Any]:
    candidates = sorted(Path("reports/eval_lab").glob("*/summary.json"))
    if not candidates:
        return {"status": "none"}
    obj = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return {"status": "ok", "audit": obj.get("audit", {})}
