from __future__ import annotations

import os
from pathlib import Path

SAFE_SQLITE_FILES = ("stockvn.db", "dev.db", "ci_sqlite.db")


def main() -> int:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./stockvn.db")
    if not database_url.startswith("sqlite:///"):
        print("Refusing reset: only sqlite DATABASE_URL is allowed for safe reset.")
        return 1

    db_path = Path(database_url.replace("sqlite:///", ""))
    if db_path.name not in SAFE_SQLITE_FILES:
        print(f"Refusing reset: {db_path} is not in allow-list {SAFE_SQLITE_FILES}.")
        return 1

    if db_path.exists():
        db_path.unlink()
        print(f"Deleted database file: {db_path}")
    else:
        print(f"Database file does not exist, nothing to delete: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
