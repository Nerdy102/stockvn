from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel

from api_fastapi.main import create_app
from core.db.models import PriceOHLCV, Ticker
from core.db.session import get_engine


def test_prices_endpoint_bounded_default_range() -> None:
    app = create_app()
    client = TestClient(app)

    engine = get_engine("sqlite:///./vn_invest.db")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.merge(Ticker(symbol="AAA", name="AAA", exchange="HOSE", sector="Tech", industry="Software"))
        for i in range(500):
            ts = dt.datetime.utcnow() - dt.timedelta(days=500 - i)
            s.merge(
                PriceOHLCV(
                    symbol="AAA",
                    timeframe="1D",
                    timestamp=ts,
                    open=10,
                    high=11,
                    low=9,
                    close=10,
                    volume=1000,
                    value_vnd=10000,
                )
            )
        s.commit()

    r = client.get("/prices", params={"symbol": "AAA", "timeframe": "1D"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) <= 400


def test_tickers_pagination() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/tickers", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    assert len(r.json()) <= 5
