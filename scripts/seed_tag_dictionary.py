from __future__ import annotations

from api_fastapi.routers.watchlists import seed_tag_dictionary
from core.db.session import create_db_and_tables, get_engine
from core.settings import get_settings
from sqlmodel import Session


def main() -> None:
    settings = get_settings()
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    with Session(engine) as session:
        inserted = seed_tag_dictionary(session)
    print(f"seed_tag_dictionary inserted={inserted}")


if __name__ == "__main__":
    main()
