import json
from pathlib import Path

import pandas as pd


def test_improvement_suite_scoreboards_exist() -> None:
    base = Path("reports/improvements")
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "improvement_summary.json").exists()]
    assert runs
    latest = sorted(runs, key=lambda p: p.stat().st_mtime)[-1]

    payload = json.loads((latest / "improvement_summary.json").read_text(encoding="utf-8"))
    assert "objective_weights" in payload
    assert "dev_scoreboard" in payload
    assert "lockbox_scoreboard" in payload

    dev = latest / "dev_scoreboard.csv"
    lock = latest / "lockbox_scoreboard.csv"
    assert dev.exists()
    assert lock.exists()

    dev_df = pd.read_csv(dev)
    lock_df = pd.read_csv(lock)
    assert "strategy" in dev_df.columns
    assert "strategy" in lock_df.columns
    assert "objective_score" in dev_df.columns
    assert "objective_score" in lock_df.columns
