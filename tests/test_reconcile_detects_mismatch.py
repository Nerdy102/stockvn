from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient
from sqlmodel import Session

from api_fastapi.main import app
from core.db.session import get_engine
from core.oms.models import Order, PortfolioSnapshot
from core.settings import get_settings


def test_reconcile_detects_mismatch() -> None:
    settings = get_settings()
    engine = get_engine(settings.DATABASE_URL)
    with Session(engine) as s:
        s.add(
            Order(
                user_id='u-mis',
                market='vn',
                symbol='FPT',
                side='BUY',
                qty=100,
                price=10000,
                status='FILLED',
                idempotency_key=f'mis-{dt.datetime.utcnow().isoformat()}',
            )
        )
        s.add(
            PortfolioSnapshot(
                ts=dt.datetime.utcnow(),
                cash=1000.0,
                positions_json={'FPT': 10_000.0},
                nav_est=1.0,
                drawdown_est=0.0,
            )
        )
        s.commit()

    c = TestClient(app)
    run = c.post('/reconcile/run')
    assert run.status_code == 200
    latest = c.get('/reconcile/latest')
    assert latest.status_code == 200
    assert latest.json()['status'] == 'MISMATCH'
    assert latest.json()['mismatch_count'] > 0
