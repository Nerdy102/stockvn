from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "bar_builder"))

from bar_builder.bar_builder import SessionAwareBarBuilder


def _events() -> list[dict[str, object]]:
    rows = []
    for line in (
        Path("tests/fixtures/market_events_fixture.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        rows.append(json.loads(line))
    return rows


def _build_rows(rows: list[dict[str, object]], tf: str) -> list[dict[str, object]]:
    b = SessionAwareBarBuilder(exchange="HOSE")
    out: list[dict[str, object]] = []
    for e in rows:
        finalized, _ = b.ingest_trade(
            symbol=str(e["symbol"]),
            timeframe=tf,
            provider_ts=dt.datetime.fromisoformat(str(e["provider_ts"]).replace("Z", "+00:00")),
            price=float(e["price"]),
            qty=float(e["qty"]),
            payload_hash=str(e["payload_hash"]),
            event_id=str(e["event_id"]),
        )
        out.extend(finalized)
    return out


def test_bar_hash_deterministic() -> None:
    rows = _events()
    bars15_a = _build_rows(rows, "15m")
    bars15_b = _build_rows(list(rows), "15m")
    bars60_a = _build_rows(rows, "60m")
    bars60_b = _build_rows(list(rows), "60m")

    assert [str(x["bar_hash"]) for x in bars15_a] == [str(x["bar_hash"]) for x in bars15_b]
    assert [str(x["bar_hash"]) for x in bars60_a] == [str(x["bar_hash"]) for x in bars60_b]

    import pandas as pd

    exp15 = pd.read_csv("tests/golden/bars_expected_15m.csv")
    exp60 = pd.read_csv("tests/golden/bars_expected_60m.csv")
    got15 = pd.DataFrame(bars15_a)[exp15.columns]
    got60 = pd.DataFrame(bars60_a)[exp60.columns]
    got15["lineage_payload_hashes_json"] = got15["lineage_payload_hashes_json"].astype(str)
    got60["lineage_payload_hashes_json"] = got60["lineage_payload_hashes_json"].astype(str)
    exp15["lineage_payload_hashes_json"] = exp15["lineage_payload_hashes_json"].astype(str)
    exp60["lineage_payload_hashes_json"] = exp60["lineage_payload_hashes_json"].astype(str)
    pd.testing.assert_frame_equal(
        got15.reset_index(drop=True), exp15.reset_index(drop=True), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        got60.reset_index(drop=True), exp60.reset_index(drop=True), check_dtype=False
    )
