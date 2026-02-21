from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from sqlmodel import Session

from core.db.session import create_db_and_tables, get_database_url, get_engine
from data.corporate_actions.models import CorporateAction


def main() -> None:
    db_url = get_database_url()
    create_db_and_tables(db_url)
    engine = get_engine(db_url)

    base = Path("data_drop/corporate_actions")
    files = sorted(base.glob("*.csv"))
    if not files:
        print("Không có file corporate actions trong data_drop/corporate_actions")
        return

    with Session(engine) as session:
        inserted = 0
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    rec = CorporateAction(
                        symbol=str(row["symbol"]).upper(),
                        ex_date=dt.date.fromisoformat(str(row["ex_date"])),
                        action_type=str(row["action_type"]).strip(),
                        amount=float(row["amount"]),
                        note=str(row.get("note", "")),
                    )
                    session.add(rec)
                    inserted += 1
        session.commit()
    print(f"Đã nạp {inserted} bản ghi corporate actions")


if __name__ == "__main__":
    main()
