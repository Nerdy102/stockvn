import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "decision_date",
    "exec_date",
    "symbol",
    "score_raw",
    "score_used",
    "mu",
    "uncert",
    "regime",
    "prev_w",
    "target_w",
    "delta_w",
    "side",
    "exec_price",
    "est_cost_bps",
    "realized_cost_bps",
    "realized_return_next",
    "notes",
]


def test_model_outputs_schema_latest_run() -> None:
    base = Path("reports/eval_lab")
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    assert runs
    latest = sorted(runs, key=lambda p: p.stat().st_mtime)[-1]
    summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    chosen = str(summary.get("chosen_default", "USER_V0"))

    for strategy in ["USER_V0", chosen]:
        fp = latest / "model_outputs" / f"{strategy}.csv"
        assert fp.exists()
        df = pd.read_csv(fp)
        assert list(df.columns) == REQUIRED_COLUMNS
