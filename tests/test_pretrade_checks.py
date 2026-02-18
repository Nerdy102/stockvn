from __future__ import annotations

import datetime as dt
import uuid

from core.db.models import Portfolio, PriceOHLCV, Ticker
from core.db.session import create_db_and_tables, get_engine
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.oms.order_validator import validate_pretrade
from sqlmodel import Session


def test_pretrade_checks_fail_on_max_single_and_lot() -> None:
    db_url = f"sqlite:///./artifacts/test_pretrade_checks_{uuid.uuid4().hex}.db"
    create_db_and_tables(db_url)
    engine = get_engine(db_url)
    with Session(engine) as s:
        s.add(Portfolio(id=1, name="P1"))
        s.add(Ticker(symbol="AAA", name="A", exchange="HOSE", sector="Tech", industry="IT"))
        s.add(
            PriceOHLCV(
                symbol="AAA",
                timeframe="1D",
                timestamp=dt.datetime(2025, 1, 2, 0, 0),
                open=10000,
                high=10000,
                low=10000,
                close=10000,
                volume=1_000_000,
                value_vnd=20_000_000_000,
            )
        )
        s.commit()

        mr = MarketRules.from_yaml("configs/market_rules_vn.yaml")
        fees = FeesTaxes.from_yaml("configs/fees_taxes.yaml")
        out = validate_pretrade(
            session=s,
            portfolio_id=1,
            symbol="AAA",
            side="BUY",
            quantity=99_950,  # not board lot aligned and large single-name
            price=10000,
            as_of=dt.datetime(2025, 1, 2, 9, 30),
            market_rules=mr,
            fees=fees,
            broker_name="demo_broker",
        )
        assert out["ok"] is False
        assert "invalid_lot" in out["reasons"]
        assert "max_single_nav_breach" in out["reasons"]
