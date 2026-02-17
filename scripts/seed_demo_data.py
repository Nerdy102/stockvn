from __future__ import annotations

import logging

from sqlmodel import Session, select

from core.db.models import Ticker
from core.db.session import create_db_and_tables, get_engine
from core.logging import setup_logging
from core.settings import get_settings
from data.providers.factory import get_provider
from worker_scheduler.jobs import ensure_seeded

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
log = logging.getLogger(__name__)


def main() -> None:
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    provider = get_provider(settings)

    with Session(engine) as session:
        ensure_seeded(session, provider, settings)

    log.info("Seed completed.")


if __name__ == "__main__":
    main()
