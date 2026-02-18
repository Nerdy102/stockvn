from __future__ import annotations

import tempfile
from pathlib import Path

from scripts import ui_guardrail_check


def test_ui_guardrail_script_fails_for_forbidden_phrase(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        app_dir = root / "apps" / "dashboard_streamlit"
        app_dir.mkdir(parents=True)
        (app_dir / "bad.py").write_text("# guaranteed return", encoding="utf-8")
        ui_dir = app_dir / "ui"
        ui_dir.mkdir(parents=True)
        (ui_dir / "layout.py").write_text(
            "from apps.dashboard_streamlit.ui.text import DISCLAIMER_SHORT", encoding="utf-8"
        )

        monkeypatch.chdir(root)
        rc = ui_guardrail_check.main()
        assert rc == 1
