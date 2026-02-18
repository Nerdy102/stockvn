from __future__ import annotations

from core.db.session import create_db_and_tables, get_database_url, get_engine
from sqlmodel import Session
from worker_scheduler.jobs import job_train_alpha_v3


def main() -> None:
    db_url = get_database_url()
    create_db_and_tables(db_url)
    engine = get_engine(db_url)
    with Session(engine) as session:
        result = job_train_alpha_v3(session)
    print(result)


if __name__ == "__main__":
    main()
