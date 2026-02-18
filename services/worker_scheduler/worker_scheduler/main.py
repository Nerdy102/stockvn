from __future__ import annotations

import argparse

from apscheduler.schedulers.blocking import BlockingScheduler
from core.db.models import Ticker
from core.db.session import create_db_and_tables, get_engine
from core.logging import get_logger, setup_logging
from core.settings import get_settings
from data.providers.factory import get_provider
from sqlmodel import Session, select

from worker_scheduler.jobs import (
    bronze_retention_cleanup,
    cleanup_stream_dedup_job,
    compute_data_quality_metrics_job,
    compute_daily_flow_features,
    compute_daily_orderbook_features,
    compute_drift_metrics_job,
    compute_factor_scores,
    compute_indicators,
    compute_daily_intraday_features,
    compute_technical_setups,
    consume_ssi_stream_to_bronze_silver,
    ensure_partitions_monthly,
    ensure_seeded,
    generate_alerts,
    ingest_prices_job,
)

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
log = get_logger(__name__)


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
        job_log = get_logger(__name__, job_id="ingest_prices")
        with Session(engine) as session:
            ingest_prices_job(session, provider)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})

    def job_compute() -> None:
        job_log = get_logger(__name__, job_id="compute_all")
        with Session(engine) as session:
            compute_indicators(session)
            compute_factor_scores(session)
            compute_technical_setups(session)
            generate_alerts(session)
            compute_data_quality_metrics_job(session)
            compute_daily_flow_features(session)
            compute_daily_orderbook_features(session)
            compute_daily_intraday_features(session)
            compute_drift_metrics_job(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})

    def job_consume_stream() -> None:
        job_log = get_logger(__name__, job_id="consume_ssi_stream")
        with Session(engine) as session:
            consume_ssi_stream_to_bronze_silver(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})

    def job_bronze_cleanup() -> None:
        job_log = get_logger(__name__, job_id="bronze_retention_cleanup")
        with Session(engine) as session:
            bronze_retention_cleanup(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})

    def job_ensure_partitions() -> None:
        job_log = get_logger(__name__, job_id="ensure_partitions_monthly")
        with Session(engine) as session:
            created = ensure_partitions_monthly(session)
        job_log.info(
            "worker_job_completed", extra={"event": "worker_job", "created_partitions": created}
        )

    def job_cleanup_dedup() -> None:
        job_log = get_logger(__name__, job_id="cleanup_stream_dedup")
        with Session(engine) as session:
            cleanup_stream_dedup_job(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})

    if args.once:
        job_ingest()
        job_compute()
        job_consume_stream()
        job_ensure_partitions()
        job_cleanup_dedup()
        job_bronze_cleanup()
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
    scheduler.add_job(
        job_consume_stream,
        "interval",
        seconds=5,
        id="consume_ssi_stream",
        replace_existing=True,
    )
    scheduler.add_job(
        job_ensure_partitions,
        "cron",
        day=1,
        hour=0,
        minute=1,
        id="ensure_partitions_monthly",
        replace_existing=True,
    )
    scheduler.add_job(
        job_cleanup_dedup,
        "cron",
        hour=0,
        minute=5,
        id="cleanup_stream_dedup",
        replace_existing=True,
    )
    scheduler.add_job(
        job_bronze_cleanup,
        "cron",
        hour=0,
        minute=10,
        id="bronze_retention_cleanup",
        replace_existing=True,
    )

    log.info("worker_started", extra={"event": "worker_startup"})
    scheduler.start()


if __name__ == "__main__":
    main()
