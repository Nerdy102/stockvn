from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from sqlmodel import Session

from core.db.models import UniverseSnapshot
from core.db.session import create_db_and_tables, get_database_url, get_engine


def main() -> None:
    db_url = get_database_url()
    create_db_and_tables(db_url)
    engine = get_engine(db_url)

    base = Path("data_drop/universe_snapshots")
    files = sorted(base.glob("*.csv"))
    if not files:
        print("Không có file universe snapshot trong data_drop/universe_snapshots")
        return

    with Session(engine) as session:
        n = 0
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    syms = [s.strip().upper() for s in str(row.get("symbols", "")).split(",") if s.strip()]
                    session.add(
                        UniverseSnapshot(
                            universe_name=str(row["universe_name"]).upper(),
                            snapshot_date=dt.date.fromisoformat(str(row["snapshot_date"])),
                            symbols_json={"symbols": syms},
                        )
                    )
                    n += 1
        session.commit()
    print(f"Đã nạp {n} snapshot")


if __name__ == "__main__":
    main()
