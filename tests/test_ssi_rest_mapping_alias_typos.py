from __future__ import annotations

import json
from pathlib import Path

from data.providers.ssi_fastconnect.mapper_rest import map_daily_stock_price


def test_ssi_rest_mapping_alias_typos() -> None:
    payload = json.loads(Path("tests/fixtures/ssi_fcdata/rest_daily_stock_price.json").read_text())
    bars, meta = map_daily_stock_price(payload, source="ssi_fastconnect_rest")

    assert len(bars) == 1
    assert bars[0].symbol == "FPT"
    assert meta[0]["foreign_sell_value"] == 1200000000.0
    assert meta[0]["net_foreign_volume"] == 10000.0
