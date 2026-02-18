from __future__ import annotations

import datetime as dt
import uuid
import json
from pathlib import Path

from core.db.models import GovernanceState, Portfolio, PriceOHLCV, Ticker, Trade, DataHealthIncident
from core.db.session import create_db_and_tables, get_engine
from core.fees_taxes import FeesTaxes
from core.reconciliation.reconcile import reconcile_portfolio
from sqlmodel import Session, select


def test_reconcile_mismatch_creates_incident() -> None:
    db_url = f"sqlite:///./artifacts/test_reconcile_mismatch_{uuid.uuid4().hex}.db"
    create_db_and_tables(db_url)
    engine = get_engine(db_url)
    with Session(engine) as s:
        s.add(Portfolio(id=1, name="P1"))
        s.add(Ticker(symbol="AAA", name="A", exchange="HOSE", sector="Tech", industry="IT"))
        s.add(
            PriceOHLCV(
                symbol="AAA",
                timeframe="1D",
                timestamp=dt.datetime(2025, 1, 2),
                open=10000,
                high=10000,
                low=10000,
                close=10000,
                volume=1000,
                value_vnd=10_000_000,
            )
        )
        s.add(
            Trade(
                portfolio_id=1,
                trade_date=dt.date(2025, 1, 2),
                symbol="AAA",
                side="BUY",
                quantity=100,
                price=10000,
                external_id="x1",
            )
        )
        s.commit()

        fees = FeesTaxes.from_yaml("configs/fees_taxes.yaml")
        out = reconcile_portfolio(
            s,
            portfolio_id=1,
            broker_name="demo_broker",
            fees=fees,
            expected_equity=123.0,
            tolerance_vnd=1.0,
        )
        assert out["status"] == "MISMATCH"
        inc = s.exec(select(DataHealthIncident).order_by(DataHealthIncident.id.desc())).first()
        assert inc is not None
        gov = s.exec(select(GovernanceState).order_by(GovernanceState.id.desc())).first()
        assert gov is not None
        assert gov.status == "PAUSED"

        expected = json.loads(Path("tests/golden/reconcile_diff_example.json").read_text(encoding="utf-8"))
        assert expected["status"] == out["status"]
