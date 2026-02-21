from pathlib import Path
import subprocess


def test_chat_printers_run() -> None:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        ".:services:packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps"
    )
    run = subprocess.run(
        ["python", "scripts/print_eval_chat.py"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "EVAL_LAB CHAT SUMMARY" in run.stdout
    assert "CHOSEN_DEFAULT=" in run.stdout
    assert "WHY_CHOSEN=" in run.stdout
    # printed when improvement summary exists
    if "LOCKBOX VERIFY SUMMARY" in run.stdout:
        assert "FINAL_DEFAULT=" in run.stdout

    run2 = subprocess.run(
        ["python", "scripts/print_model_snapshot_chat.py", "--days", "1", "--topk", "2"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "MODEL OUTPUT SNAPSHOT" in run2.stdout
    assert "USER_V0" in run2.stdout
    assert Path("reports/eval_lab").exists()
