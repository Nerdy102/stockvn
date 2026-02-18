from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "services" / "realtime_signal_engine"))

from realtime_signal_engine.storage import SignalStorage


def test_intraday_cooldown_26_bars() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        store = SignalStorage(s)
        assert store.cooldown_allows("AAA", "15m", "expr", 0, 26)
        store.mark_cooldown("AAA", "15m", "expr", 0)
        assert not store.cooldown_allows("AAA", "15m", "expr", 25, 26)
        assert store.cooldown_allows("AAA", "15m", "expr", 26, 26)
