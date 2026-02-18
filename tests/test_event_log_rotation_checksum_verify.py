from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "market_gateway"))

from market_gateway.event_log import EventLogWriter, verify_event_log


def test_event_log_rotation_checksum_verify(tmp_path) -> None:
    writer = EventLogWriter(str(tmp_path), rotate_every=2)
    p1 = writer.append({"a": 1})
    assert p1 is None
    p2 = writer.append({"a": 2})
    assert p2 is not None
    result = verify_event_log(p2)
    assert result["ok"] is True
    assert result["rows"] == 2
