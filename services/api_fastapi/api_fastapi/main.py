from __future__ import annotations

import logging

from core.db.models import Ticker
from core.db.session import create_db_and_tables, get_engine
from core.logging import setup_logging
from core.settings import get_settings
from data.providers.factory import get_provider
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from worker_scheduler.jobs import ensure_seeded

from api_fastapi.routers import (
    alerts,
    fundamentals,
    health,
    portfolio,
    prices,
    screeners,
    signals,
    tickers,
)

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="VN Invest Toolkit API",
        version="0.1.0",
        description="Offline demo + production scaffold (educational; not investment advice).",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(tickers.router)
    app.include_router(prices.router)
    app.include_router(fundamentals.router)
    app.include_router(screeners.router)
    app.include_router(signals.router)
    app.include_router(portfolio.router)
    app.include_router(alerts.router)

    @app.on_event("startup")
    def _startup() -> None:
        create_db_and_tables(settings.DATABASE_URL)
        engine = get_engine(settings.DATABASE_URL)
        with Session(engine) as session:
            if session.exec(select(Ticker)).first() is None:
                provider = get_provider(settings)
                ensure_seeded(session, provider, settings)

    return app


app = create_app()
