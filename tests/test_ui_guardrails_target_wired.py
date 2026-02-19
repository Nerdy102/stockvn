from __future__ import annotations

import subprocess
import sys


def test_ui_guardrails_module_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ui_guardrail_check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
