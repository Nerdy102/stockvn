from __future__ import annotations

import hashlib
from pathlib import Path

from core.db.models import BronzeFile
from data.bronze.writer import BronzeWriter
from sqlmodel import Session, SQLModel, create_engine, select


def test_bronze_writer_creates_zst_and_db_row(tmp_path) -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        writer = BronzeWriter(
            provider="ssi_fastconnect",
            channel="ssi_stream_B",
            session=session,
            base_dir=str(tmp_path / "data_lake" / "bronze"),
        )
        writer.write({"k": 1})
        writer.flush()

        db_row = session.exec(select(BronzeFile)).one()
        path = Path(db_row.filepath)
        assert path.exists()
        assert path.is_absolute()

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == db_row.sha256

        data = path.read_bytes()
        try:
            import zstandard as zstd

            decoded = zstd.ZstdDecompressor().decompress(data).decode("utf-8")
        except Exception:
            decoded = data.decode("utf-8")
        assert '"k":1' in decoded
        assert db_row.rows == 1
