import hashlib
import json

from scripts.run_raocmoe_backtest import run


def test_end_to_end_smoke() -> None:
    out = run(["BTCUSDT", "ETHUSDT"], "2024-01-01", "2024-01-20")
    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {"dataset_hash", "config_hash", "code_hash"}.issubset(report)
    payload = report_path.read_text(encoding="utf-8")
    assert len(hashlib.sha256(payload.encode("utf-8")).hexdigest()) == 64
    assert (out / "equity.csv").exists()
