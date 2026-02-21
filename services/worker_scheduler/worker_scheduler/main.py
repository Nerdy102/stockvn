from __future__ import annotations

import argparse

from apscheduler.schedulers.blocking import BlockingScheduler
from core.monitoring.prometheus_metrics import start_metrics_http_server
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
    job_build_labels_v3,
    job_build_ml_features_v3,
    nightly_parquet_export_job,
    compute_factor_scores,
    compute_indicators,
    compute_daily_intraday_features,
    compute_technical_setups,
    consume_ssi_stream_to_bronze_silver,
    ensure_partitions_monthly,
    ensure_seeded,
    generate_alerts,
    ingest_prices_job,
    job_alert_sla_escalation_daily,
    job_alert_digest_daily,
    job_run_pending_interactive_runs,
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

    start_metrics_http_server(9001)

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
            compute_daily_flow_features(session)
            compute_daily_orderbook_features(session)
            compute_daily_intraday_features(session)
            job_build_labels_v3(session)
            job_build_ml_features_v3(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})


    def job_compute_data_quality() -> None:
        job_log = get_logger(__name__, job_id="compute_data_quality")
        with Session(engine) as session:
            compute_data_quality_metrics_job(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job"})

    def job_compute_drift() -> None:
        job_log = get_logger(__name__, job_id="compute_drift")
        with Session(engine) as session:
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


    def job_interactive_runs() -> None:
        job_log = get_logger(__name__, job_id="interactive_runs")
        with Session(engine) as session:
            processed = job_run_pending_interactive_runs(session)
        if processed:
            job_log.info("worker_job_completed", extra={"event": "worker_job", "processed": processed})

    def job_alert_ops() -> None:
        job_log = get_logger(__name__, job_id="alerts_ops")
        with Session(engine) as session:
            out1 = job_alert_sla_escalation_daily(session)
            out2 = job_alert_digest_daily(session, settings)
        job_log.info("worker_job_completed", extra={"event": "worker_job", **out1, **out2})

    def job_nightly_parquet_export() -> None:
        job_log = get_logger(__name__, job_id="nightly_parquet_export")
        with Session(engine) as session:
            out = nightly_parquet_export_job(session)
        job_log.info("worker_job_completed", extra={"event": "worker_job", **out})

    if args.once:
        job_ingest()
        job_compute()
        job_compute_data_quality()
        job_compute_drift()
        job_consume_stream()
        job_ensure_partitions()
        job_cleanup_dedup()
        job_nightly_parquet_export()
        job_alert_ops()
        job_bronze_cleanup()
        job_interactive_runs()
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
        job_compute_data_quality,
        "cron",
        hour=0,
        minute=20,
        id="compute_data_quality_daily",
        replace_existing=True,
    )
    scheduler.add_job(
        job_compute_drift,
        "cron",
        day_of_week="mon",
        hour=0,
        minute=30,
        id="compute_drift_weekly",
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
        job_nightly_parquet_export,
        "cron",
        hour=1,
        minute=10,
        id="nightly_parquet_export",
        replace_existing=True,
    )
    scheduler.add_job(
        job_alert_ops,
        "cron",
        hour=11,
        minute=5,
        id="alerts_ops",
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
    scheduler.add_job(
        job_interactive_runs,
        "interval",
        seconds=3,
        id="interactive_runs",
        replace_existing=True,
        max_instances=1,
    )

    log.info("worker_started", extra={"event": "worker_startup"})
    scheduler.start()


if __name__ == "__main__":
    main()
