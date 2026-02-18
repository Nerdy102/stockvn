from __future__ import annotations

from pathlib import Path


def test_migration_0002_has_table_and_index_guards() -> None:
    content = Path("migrations/versions/20260218_0002_daily_feature_tables.py").read_text()
    assert "def _has_table(" in content
    assert "def _has_index(" in content
    assert "if not _has_table(inspector, \"daily_flow_features\")" in content
    assert "if not _has_table(inspector, \"daily_orderbook_features\")" in content
    assert "if not _has_table(inspector, \"daily_intraday_features\")" in content
    assert "if not _has_table(inspector, \"feature_last_processed\")" in content
    assert "if _has_table(inspector, \"market_daily_meta\")" in content
