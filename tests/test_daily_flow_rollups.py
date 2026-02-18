from __future__ import annotations

import datetime as dt

from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DailyFlowFeature, MarketDailyMeta, PriceOHLCV
from core.features.daily_flow import compute_daily_flow_features


def test_daily_flow_rollups_5d_20d() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        base = dt.datetime(2024, 1, 1, 15, 0)
        for i in range(20):
            ts = base + dt.timedelta(days=i)
            s.add(
                MarketDailyMeta(
                    symbol="AAA",
                    timestamp=ts,
                    foreign_buy_value=100.0,
                    foreign_sell_value=40.0,
                    current_room=80.0,
                    total_room=100.0,
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

        up = compute_daily_flow_features(s)
        assert up == 20

        row = s.exec(
            select(DailyFlowFeature)
            .where(DailyFlowFeature.symbol == "AAA")
            .where(DailyFlowFeature.date == dt.date(2024, 1, 20))
            .where(DailyFlowFeature.source == "stream_r")
        ).first()
        assert row is not None
        assert row.net_foreign_val_day == 60.0
        assert row.net_foreign_val_5d == 300.0
        assert row.net_foreign_val_20d == 1200.0
        assert row.foreign_flow_intensity == 12.0
        assert abs((row.foreign_room_util or 0.0) - 0.2) < 1e-12
