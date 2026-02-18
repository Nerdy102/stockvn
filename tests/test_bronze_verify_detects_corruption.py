from __future__ import annotations

from scripts.bronze_verify import verify_bronze_files


def test_bronze_verify_detects_corruption(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from core.db.models import BronzeFile
    from core.db.session import get_engine
    from data.bronze.writer import BronzeWriter
    from sqlmodel import Session, SQLModel, select

    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        writer = BronzeWriter(
            "ssi_fastconnect", "ssi_stream_B", session=session, base_dir=str(tmp_path)
        )
        writer.write({"x": 1})
        writer.flush()
        row = session.exec(select(BronzeFile)).one()

        with open(row.filepath, "rb") as f:
            data = f.read()
        with open(row.filepath, "wb") as f:
            f.write(data[: max(1, len(data) // 2)])

    corrupted = verify_bronze_files(f"sqlite:///{db_path}")
    assert corrupted == 1
