import json
from pathlib import Path


def test_eval_summary_has_chosen_default() -> None:
    base = Path("reports/eval_lab")
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    assert runs
    latest = sorted(runs, key=lambda p: p.stat().st_mtime)[-1]
    payload = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    assert "chosen_default" in payload
    assert isinstance(payload["chosen_default"], str)
    assert payload["chosen_default"].startswith("USER_V")
