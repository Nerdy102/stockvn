from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "data"))

from lakehouse.bronze import append_bronze_records, read_bronze_partition
from lakehouse.silver import bronze_to_canonical_trades


def _load_fixture() -> list[dict[str, object]]:
    rows = []
    for line in (
        Path("tests/fixtures/bronze_payloads_tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        rows.append(json.loads(line))
    return rows


def test_bronze_to_silver_is_deterministic(tmp_path) -> None:
    base = tmp_path / "lake"
    rows = _load_fixture()
    append_bronze_records(base, rows)
    append_bronze_records(base, rows)

    bronze = read_bronze_partition(
        base, dt_value=dt.date(2025, 1, 2), source="demo", channel="trade"
    )
    assert len(bronze) == 2

    silver_a = bronze_to_canonical_trades(bronze)
    silver_b = bronze_to_canonical_trades(bronze.sample(frac=1.0, random_state=42))

    pd.testing.assert_frame_equal(silver_a.reset_index(drop=True), silver_b.reset_index(drop=True))

    expected = pd.read_csv("tests/golden/bronze_to_silver_expected_trades.csv")
    silver_cmp = silver_a[expected.columns].copy().reset_index(drop=True)
    pd.testing.assert_frame_equal(silver_cmp, expected.reset_index(drop=True), check_dtype=False)
