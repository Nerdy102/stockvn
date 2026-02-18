from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path


CHECKS: list[tuple[str, str]] = [
    ("T0_quality_gate", "make quality-gate"),
    ("T1_ui_guardrail", "python scripts/ui_guardrail_check.py"),
    (
        "T2_contract_and_bootstrap_tests",
        "PYTHONPATH=packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps pytest -q tests/test_contract_hash_stability.py tests/test_ci_forbidden_strings_guardrail.py tests/test_make_targets_exist.py tests/test_rt_harness.py",
    ),
    (
        "T3_replay_fixture",
        "PYTHONPATH=packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps python -m scripts.replay_events --fixture tests/fixtures/replay/event_log_fixture.jsonl --speed max --dry-run",
    ),
    (
        "T4_rt_load",
        "PYTHONPATH=packages/core:packages/data:packages python -m tools.realtime_harness.run_load --symbols 50 --days 1 --seed 42 --out artifacts/verification/MEGA09_load_results.json",
    ),
    (
        "T5_rt_chaos",
        "PYTHONPATH=packages/core:packages/data:packages python -m tools.realtime_harness.run_chaos --seed 42 --out artifacts/verification/MEGA09_chaos_results.json",
    ),
]


def run() -> int:
    art = Path("artifacts/verification")
    art.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str | int]] = []
    for name, cmd in CHECKS:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        (art / f"{name}.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        results.append({"name": name, "cmd": cmd, "returncode": proc.returncode})
        if proc.returncode != 0:
            break

    status = (
        "pass"
        if all(int(r["returncode"]) == 0 for r in results) and len(results) == len(CHECKS)
        else "fail"
    )
    report = {
        "status": status,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "checks": results,
    }
    (art / "PROGRAM_VERIFICATION_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = ["# Program Verification Report", "", f"- Status: {status.upper()}", "", "## Checks"]
    for r in results:
        mark = "PASS" if int(r["returncode"]) == 0 else "FAIL"
        lines.append(f"- {r['name']}: {mark} (`{r['cmd']}`)")
    (art / "PROGRAM_VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run())
