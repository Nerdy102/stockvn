from __future__ import annotations

from pathlib import Path

from scripts.replay_smoke import run_smoke


def test_replay_parity_integration_fixture_window_passes(tmp_path: Path, monkeypatch) -> None:
    del tmp_path
    monkeypatch.chdir(Path.cwd())
    # run_smoke loads 2-day fixture and enforces parity tolerances
    run_smoke()
