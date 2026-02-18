from __future__ import annotations

import json
from pathlib import Path

from api_fastapi.routers import screeners


def test_validation_normalizes_and_hash_stable() -> None:
    payload = {
        "name": "demo",
        "as_of_date": "2025-01-10",
        "universe": {"preset": "ALL"},
        "filters": {
            "min_adv20_value": 1_000_000_000.0,
            "sector_in": ["Tech", "Bank", "Tech"],
            "exchange_in": ["HOSE", "HOSE"],
            "tags_any": ["Policy", "policy"],
            "tags_all": ["KQKD"],
            "neutralization": {"enabled": False},
        },
        "factor_weights": {"quality": 2, "value": 2, "momentum": 2, "lowvol": 2, "dividend": 2},
        "technical_setups": {
            "breakout": True,
            "trend": True,
            "pullback": False,
            "volume_spike": False,
        },
    }
    out = screeners._canonicalize_and_validate(payload)
    assert out.errors == []

    expected = json.loads(
        Path("tests/golden/screen_v4_normalized.json").read_text(encoding="utf-8")
    )
    norm = {k: v for k, v in out.normalized.items() if k != "_warnings"}
    assert norm == expected

    h1 = screeners._hash_screen(out.normalized)
    h2 = screeners._hash_screen(out.normalized)
    assert h1 == h2
