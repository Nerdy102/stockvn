from __future__ import annotations

import argparse
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from core.db.models import Ticker
from core.db.session import create_db_and_tables, get_engine
from core.logging import setup_logging
from core.settings import get_settings
from data.providers.factory import get_provider
from sqlmodel import Session, select

from worker_scheduler.jobs import (
    compute_data_quality_metrics_job,
    compute_drift_metrics_job,
    compute_factor_scores,
    compute_indicators,
    compute_technical_setups,
    ensure_seeded,
    generate_alerts,
    ingest_prices_job,
)

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="VN Invest Toolkit Worker Scheduler")
    parser.add_argument("--once", action="store_true", help="Run jobs once and exit")
    parser.add_argument(
        "--interval-minutes", type=int, default=15, help="Interval minutes for compute jobs"
    )
    args = parser.parse_args()

    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    provider = get_provider(settings)

    with Session(engine) as session:
        if session.exec(select(Ticker)).first() is None:
            ensure_seeded(session, provider, settings)

    def job_ingest() -> None:
        with Session(engine) as session:
            ingest_prices_job(session, provider)

    def job_compute() -> None:
        with Session(engine) as session:
            compute_indicators(session)
            compute_factor_scores(session)
            compute_technical_setups(session)
            generate_alerts(session)
            compute_data_quality_metrics_job(session)
            compute_drift_metrics_job(session)

    if args.once:
        job_ingest()
        job_compute()
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        job_ingest,
        "interval",
        minutes=max(60, args.interval_minutes),
        id="ingest_prices",
        replace_existing=True,
    )
    scheduler.add_job(
        job_compute,
        "interval",
        minutes=args.interval_minutes,
        id="compute_all",
        replace_existing=True,
    )

    log.info("Worker started. ingest>=60m, compute=%sm", args.interval_minutes)
    scheduler.start()


if __name__ == "__main__":
    main()
