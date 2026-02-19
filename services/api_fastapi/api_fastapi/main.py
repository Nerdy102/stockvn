from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from core.db.models import Ticker
from core.oms import models as oms_models
from core.risk import controls_models
from core.reconciliation import models as reconcile_models
from core.db.session import create_db_and_tables, get_engine
from core.logging import get_logger, request_id_context, setup_logging
from core.monitoring.prometheus_metrics import metrics_payload
from core.settings import get_settings
from data.providers.factory import get_provider
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlmodel import Session, select
from worker_scheduler.jobs import ensure_seeded

from api_fastapi.routers import (
    alerts,
    data_health,
    chart,
    controls,
    fundamentals,
    health,
    ml,
    orders,
    oms,
    portfolio,
    prices,
    realtime,
    screeners,
    signals,
    simple_mode,
    tickers,
    universe,
    watchlists,
)

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    with Session(engine) as session:
        if session.exec(select(Ticker)).first() is None:
            provider = get_provider(settings)
            ensure_seeded(session, provider, settings)
        watchlists.seed_tag_dictionary(session)
    yield


def create_app() -> FastAPI:
    # Ensure local SQLite test/dev bootstraps always have schema, even when lifespan
    # is not entered (e.g. some TestClient usage patterns).
    create_db_and_tables(settings.DATABASE_URL)

    app = FastAPI(
        title="VN Invest Toolkit API",
        version="0.1.0",
        description="Offline demo + production scaffold (educational; not investment advice).",
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        with request_id_context(request_id):
            response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=500)

    @app.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        payload, content_type = metrics_payload()
        return Response(content=payload, media_type=content_type)

    app.include_router(health.router)
    app.include_router(tickers.router)
    app.include_router(prices.router)
    app.include_router(realtime.router)
    app.include_router(orders.router)
    app.include_router(oms.router)
    app.include_router(fundamentals.router)
    app.include_router(screeners.router)
    app.include_router(signals.router)
    app.include_router(simple_mode.router)
    app.include_router(portfolio.router)
    app.include_router(alerts.router)
    app.include_router(data_health.router)
    app.include_router(chart.router)
    app.include_router(controls.router)
    app.include_router(ml.router)
    app.include_router(universe.router)
    app.include_router(watchlists.router)

    log.info("api_initialized", extra={"event": "api_startup"})
    return app


app = create_app()
