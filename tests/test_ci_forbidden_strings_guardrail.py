from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ui_guardrail_check import main


def test_ui_guardrail_script_passes_current_ui() -> None:
    assert main() == 0
