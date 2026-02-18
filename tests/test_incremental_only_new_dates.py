from __future__ import annotations

import datetime as dt

from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DailyFlowFeature, FeatureLastProcessed, MarketDailyMeta, PriceOHLCV
from core.features.daily_flow import FEATURE_NAME, compute_daily_flow_features


def test_incremental_only_new_dates() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        d1 = dt.datetime(2024, 1, 1, 15, 0)
        d2 = dt.datetime(2024, 1, 2, 15, 0)
        for ts in (d1, d2):
            s.add(
                MarketDailyMeta(
                    symbol="AAA",
                    timestamp=ts,
                    foreign_buy_value=100.0,
                    foreign_sell_value=50.0,
                    source="stream_r",
                )
            )
            s.add(
                PriceOHLCV(
                    symbol="AAA",
                    timeframe="1D",
                    timestamp=ts,
                    open=10,
                    high=11,
                    low=9,
                    close=10,
                    volume=1_000,
                    value_vnd=100.0,
                    source="legacy",
                    quality_flags={},
                )
            )
        s.commit()

        up1 = compute_daily_flow_features(s)
        assert up1 == 2

        s.add(
            MarketDailyMeta(
                symbol="AAA",
                timestamp=dt.datetime(2024, 1, 3, 15, 0),
                foreign_buy_value=120.0,
                foreign_sell_value=50.0,
                source="stream_r",
            )
        )
        s.add(
            PriceOHLCV(
                symbol="AAA",
                timeframe="1D",
                timestamp=dt.datetime(2024, 1, 3, 15, 0),
                open=10,
                high=11,
                low=9,
                close=10,
                volume=1_000,
                value_vnd=100.0,
                source="legacy",
                quality_flags={},
            )
        )
        s.commit()

        up2 = compute_daily_flow_features(s)
        assert up2 == 1

        total = len(s.exec(select(DailyFlowFeature)).all())
        assert total == 3

        state = s.exec(
            select(FeatureLastProcessed)
            .where(FeatureLastProcessed.feature_name == FEATURE_NAME)
            .where(FeatureLastProcessed.symbol == "")
        ).first()
        assert state is not None
        assert state.last_date == dt.date(2024, 1, 3)
