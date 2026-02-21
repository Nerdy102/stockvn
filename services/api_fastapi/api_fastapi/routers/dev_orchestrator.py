from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/dev/orchestrator", tags=["dev_orchestrator"])

SERVICE_CMDS = {
    "binance_ingestor": ["python", "services/stream_ingestor/stream_ingestor/main.py"],
    "bar_builder": ["python", "services/bar_builder/bar_builder/main.py"],
    "signal_engine": ["python", "services/realtime_signal_engine/realtime_signal_engine/main.py"],
}
BASE = Path("artifacts/orchestrator")


def _ensure_enabled() -> None:
    if os.getenv("ENABLE_DEV_ORCHESTRATOR", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="ENABLE_DEV_ORCHESTRATOR=true required")


def _pid_file(service: str) -> Path:
    BASE.mkdir(parents=True, exist_ok=True)
    return BASE / f"{service}.pid"


@router.post("/start")
def start_service(service: str = Query(..., pattern="^(binance_ingestor|bar_builder|signal_engine)$")) -> dict[str, str]:
    _ensure_enabled()
    pidf = _pid_file(service)
    if pidf.exists():
        return {"status": "already_running", "service": service}
    logf = BASE / f"{service}.log"
    with logf.open("ab") as f:
        proc = subprocess.Popen(SERVICE_CMDS[service], stdout=f, stderr=f)
    pidf.write_text(str(proc.pid), encoding="utf-8")
    return {"status": "started", "service": service}


@router.post("/stop")
def stop_service(service: str = Query(..., pattern="^(binance_ingestor|bar_builder|signal_engine)$")) -> dict[str, str]:
    _ensure_enabled()
    pidf = _pid_file(service)
    if not pidf.exists():
        return {"status": "not_running", "service": service}
    pid = int(pidf.read_text(encoding="utf-8").strip())
    os.kill(pid, signal.SIGTERM)
    pidf.unlink(missing_ok=True)
    return {"status": "stopped", "service": service}


@router.get("/status")
def status() -> dict[str, dict[str, str]]:
    _ensure_enabled()
    out = {}
    for svc in SERVICE_CMDS:
        pidf = _pid_file(svc)
        out[svc] = {"running": str(pidf.exists()).lower(), "pid": pidf.read_text(encoding="utf-8").strip() if pidf.exists() else ""}
    return out
