from __future__ import annotations

from pathlib import Path
from typing import Any

from core.screener.engine import ScreenDefinition, run_screen
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from api_fastapi.deps import get_db

router = APIRouter(prefix="/screeners", tags=["screeners"])


class RunScreenRequest(BaseModel):
    screen_path: str | None = None
    inline_yaml: str | None = None


@router.get("")
def list_screens() -> dict[str, Any]:
    screens_dir = Path("configs/screens")
    screens = []
    if screens_dir.exists():
        for p in screens_dir.glob("*.yaml"):
            screens.append({"name": p.stem, "path": str(p)})
    return {"screens": screens}


@router.post("/run")
def run_screen_endpoint(payload: RunScreenRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.inline_yaml:
        tmp = Path(".tmp_screen.yaml")
        tmp.write_text(payload.inline_yaml, encoding="utf-8")
        screen = ScreenDefinition.from_yaml(tmp)
        tmp.unlink(missing_ok=True)
    else:
        screen = ScreenDefinition.from_yaml(
            payload.screen_path or "configs/screens/demo_screen.yaml"
        )

    results = run_screen(db, screen)
    return {"screen": screen.name, "description": screen.description, "results": results}
