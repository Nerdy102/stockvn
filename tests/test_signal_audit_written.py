from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlmodel import Session, select

from core.db.session import get_engine
from core.simple_mode.audit_models import SignalAudit
from core.simple_mode.models import run_signal


def test_signal_audit_written(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    rows = []
    for i in range(220):
        rows.append({
            "date": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)),
            "open": 100 + i * 0.1,
            "high": 101 + i * 0.1,
            "low": 99 + i * 0.1,
            "close": 100 + i * 0.1,
            "volume": 1_000_000,
        })
    run_signal("model_1", "FPT", "1D", pd.DataFrame(rows))
    with Session(get_engine(os.environ["DATABASE_URL"])) as s:
        audits = s.exec(select(SignalAudit)).all()
    assert len(audits) >= 1
