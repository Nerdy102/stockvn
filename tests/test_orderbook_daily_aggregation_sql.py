from __future__ import annotations

import datetime as dt

from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DailyOrderbookFeature, QuoteL2
from core.features.daily_orderbook import compute_daily_orderbook_features


def test_orderbook_daily_aggregation_sql() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    bids = {"prices": [100.0, 99.9, 99.8], "volumes": [100.0, 80.0, 70.0]}
    asks = {"prices": [100.2, 100.3, 100.4], "volumes": [50.0, 60.0, 65.0]}

    with Session(engine) as s:
        s.add(
            QuoteL2(
                symbol="AAA",
                timestamp=dt.datetime(2024, 1, 2, 9, 0),
                bids=bids,
                asks=asks,
                source="xquote",
            )
        )
        s.add(
            QuoteL2(
                symbol="AAA",
                timestamp=dt.datetime(2024, 1, 2, 9, 5),
                bids=bids,
                asks=asks,
                source="xquote",
            )
        )
        s.commit()

        up = compute_daily_orderbook_features(s)
        assert up == 1

        row = s.exec(select(DailyOrderbookFeature)).first()
        assert row is not None
        assert round(row.imb_1_day, 6) == round((100.0 - 50.0) / (150.0 + 1e-9), 6)
        assert round(row.imb_3_day, 6) == round((250.0 - 175.0) / (425.0 + 1e-9), 6)
        mid = (100.2 + 100.0) / 2.0
        assert round(row.spread_day, 6) == round((100.2 - 100.0) / mid, 6)
