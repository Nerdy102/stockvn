from __future__ import annotations

import hashlib
import os
from pathlib import Path

from core.db.models import BronzeFile
from core.db.session import create_db_and_tables, get_engine
from sqlmodel import Session, select


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bronze_files(database_url: str) -> int:
    create_db_and_tables(database_url)
    engine = get_engine(database_url)
    corrupted = 0
    with Session(engine) as session:
        files = session.exec(select(BronzeFile)).all()
        for row in files:
            path = Path(row.filepath)
            if not path.exists():
                print(f"MISSING {row.filepath}")
                corrupted += 1
                continue
            digest = _sha256_file(path)
            if digest != row.sha256:
                print(f"CORRUPT {row.filepath} expected={row.sha256} actual={digest}")
                corrupted += 1
    print(f"verified={len(files)} corrupted={corrupted}")
    return corrupted


if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL", "sqlite:///./vn_invest.db")
    raise SystemExit(1 if verify_bronze_files(db_url) else 0)
