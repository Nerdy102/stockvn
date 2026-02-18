from __future__ import annotations

import datetime as dt

from api_fastapi.main import create_app
from core.db.models import FactorScore, PriceOHLCV, Ticker, TickerLifecycle
from core.db.session import get_engine
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel


def test_explain_schema_required_keys(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "screen_explain.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    app = create_app()
    client = TestClient(app)
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    as_of = dt.date(2025, 1, 10)
    with Session(engine) as s:
        s.add(Ticker(symbol="AAA", name="AAA", exchange="HOSE", sector="Tech", industry="Soft"))
        s.add(
            TickerLifecycle(
                symbol="AAA",
                first_trading_date=dt.date(2023, 1, 1),
                last_trading_date=None,
                exchange="HOSE",
                sectype="stock",
                sector="Tech",
                source="test",
            )
        )
        for i in range(520):
            d = dt.date(2024, 1, 1) + dt.timedelta(days=i)
            if d.weekday() >= 5:
                continue
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
                )
            )
        for fac in ["value", "quality", "momentum", "lowvol", "dividend"]:
            s.add(FactorScore(symbol="AAA", as_of_date=as_of, factor=fac, score=1.0, raw={}))
        s.commit()

    screen = {
        "name": "demo",
        "as_of_date": as_of.isoformat(),
        "universe": {"preset": "ALL"},
        "filters": {
            "min_adv20_value": 1_000_000_000.0,
            "sector_in": [],
            "exchange_in": [],
            "tags_any": [],
            "tags_all": [],
            "neutralization": {"enabled": False},
        },
        "factor_weights": {
            "value": 0.2,
            "quality": 0.2,
            "momentum": 0.2,
            "lowvol": 0.2,
            "dividend": 0.2,
        },
        "technical_setups": {
            "breakout": False,
            "trend": False,
            "pullback": False,
            "volume_spike": False,
        },
    }
    r = client.post("/screeners/run", json={"screen": screen})
    assert r.status_code == 200
    explain = r.json()["results"][0]["explain"]
    assert sorted(explain.keys()) == ["factor", "filters", "final", "story", "technicals"]
