from __future__ import annotations

from core.db.session import create_db_and_tables, get_engine
from core.settings import get_settings
from sqlmodel import Session
from worker_scheduler.jobs import bronze_retention_cleanup

if __name__ == "__main__":
    settings = get_settings()
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    with Session(engine) as session:
        print(bronze_retention_cleanup(session))
