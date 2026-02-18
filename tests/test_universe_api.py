from __future__ import annotations

import datetime as dt

from core.db.models import PriceOHLCV, TickerLifecycle
from core.db.session import create_db_and_tables, get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session


def test_universe_endpoint_returns_audit_and_breakdown(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "api_universe.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    create_db_and_tables(db_url)
    engine = get_engine(db_url)
    with Session(engine) as s:
        s.add(
            TickerLifecycle(
                symbol="AAA",
                first_trading_date=dt.date(2023, 1, 1),
                last_trading_date=None,
                exchange="HOSE",
                sectype="stock",
                sector="GEN",
                source="test",
            )
        )
        trading_rows = 0
        for i in range(450):
            d = dt.date(2023, 1, 1) + dt.timedelta(days=i)
            if d.weekday() >= 5:
                continue
            trading_rows += 1
            s.add(
                PriceOHLCV(
                    symbol="AAA",
                    timeframe="1D",
                    timestamp=dt.datetime.combine(d, dt.time()),
                    open=10,
                    high=11,
                    low=9,
                    close=10,
                    volume=1000,
                    value_vnd=2_000_000_000,
                    source="test",
                    quality_flags={},
                )
            )
        assert trading_rows >= 260
        s.commit()

    from api_fastapi.main import create_app

    c = TestClient(create_app())
    r = c.get("/universe", params={"date": "05-10-2024", "name": "ALL"})
    assert r.status_code == 200
    body = r.json()
    assert "audit" in body
    assert "included_count" in body["audit"]
    assert "excluded_json_breakdown" in body["audit"]
