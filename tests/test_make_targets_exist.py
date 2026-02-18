from __future__ import annotations

from pathlib import Path


def test_required_make_targets_exist() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")
    required = {
        "quality-gate:",
        "run-api:",
        "run-worker:",
        "run-ui:",
        "run-realtime:",
        "replay-demo:",
        "verify-program:",
        "rt-load-test:",
        "rt-chaos-test:",
        "rt-verify:",
    }
    for target in required:
        assert target in text
