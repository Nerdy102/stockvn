from __future__ import annotations

import asyncio
from pathlib import Path

from core.db.models import BronzeFile
from data.providers.ssi_fastconnect.provider_rest import SsiRestProvider
from sqlmodel import Session, SQLModel, create_engine, select


class _FakeRestClient:
    async def request(self, method: str, path: str, params=None):
        if path == "/Securities":
            return [{"Symbol": "AAA", "Market": "HOSE"}]
        if path == "/SecuritiesDetails":
            return {
                "RepeatedInfo": [
                    {
                        "Symbol": "AAA",
                        "Market": "HOSE",
                        "Sector": "Tech",
                        "Industry": "Software",
                    }
                ]
            }
        raise AssertionError(f"unexpected path {path}")


def test_ssi_rest_provider_writes_bronze_file(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        provider = SsiRestProvider(
            base_url="https://example.com",
            client=_FakeRestClient(),
            session=session,
            bronze_dir=str(tmp_path),
        )

        asyncio.run(provider.get_tickers())

        files = session.exec(select(BronzeFile)).all()
        assert len(files) == 2
        assert all("ssi_rest_" in file.channel for file in files)
        assert all(Path(file.filepath).exists() for file in files)
