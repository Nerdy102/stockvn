from __future__ import annotations

from pathlib import Path

from scripts.quality_gate import find_violations


def test_quality_gate_ignores_fixtures_but_fails_packages(tmp_path: Path) -> None:
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "packages" / "core").mkdir(parents=True)

    fixture_file = tmp_path / "tests" / "fixtures" / "fixture_with_todo.py"
    fixture_file.write_text("# TO" "DO: this fixture is allowed\n", encoding="utf-8")

    package_file = tmp_path / "packages" / "core" / "bad_placeholder.py"
    package_file.write_text("# TO" "DO: this must fail\n", encoding="utf-8")

    violations = find_violations(tmp_path)

    assert all("tests/fixtures" not in violation.path for violation in violations)
    assert any(violation.path.endswith("packages/core/bad_placeholder.py") for violation in violations)
