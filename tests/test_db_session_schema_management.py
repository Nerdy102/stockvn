from __future__ import annotations

from unittest.mock import patch

from core.db.session import create_db_and_tables


def test_create_db_and_tables_runs_for_sqlite() -> None:
    with patch("core.db.session.SQLModel.metadata.create_all") as create_all:
        create_db_and_tables("sqlite:///:memory:")
    create_all.assert_called_once()


def test_create_db_and_tables_skips_for_postgres() -> None:
    with patch("core.db.session.SQLModel.metadata.create_all") as create_all:
        create_db_and_tables("postgresql+psycopg2://postgres:postgres@localhost:5432/stockvn")
    create_all.assert_not_called()
