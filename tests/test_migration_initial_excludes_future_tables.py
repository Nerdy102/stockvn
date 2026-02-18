from __future__ import annotations

from pathlib import Path


def test_initial_migration_excludes_future_tables() -> None:
    content = Path("migrations/versions/20260218_0001_initial_schema.py").read_text()
    assert "FUTURE_TABLES_EXCLUDED_FROM_INITIAL" in content
    assert '"daily_flow_features"' in content
    assert '"daily_orderbook_features"' in content
    assert '"daily_intraday_features"' in content
    assert '"feature_last_processed"' in content
