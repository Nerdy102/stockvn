from __future__ import annotations

import datetime as dt

from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DailyFlowFeature, MarketDailyMeta, PriceOHLCV
from core.features.daily_flow import compute_daily_flow_features


def test_daily_flow_uses_single_latest_room_snapshot_with_tied_timestamps() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        ts = dt.datetime(2024, 1, 10, 15, 0)
        # two rows same day should still generate exactly one feature row
        s.add(
            MarketDailyMeta(
                symbol="AAA",
                timestamp=ts,
                foreign_buy_value=10.0,
                foreign_sell_value=3.0,
                current_room=80.0,
                total_room=100.0,
                source="stream_r",
            )
        )
        s.add(
            MarketDailyMeta(
                symbol="AAA",
                timestamp=ts + dt.timedelta(minutes=1),
                foreign_buy_value=20.0,
                foreign_sell_value=4.0,
                current_room=70.0,
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
                volume=100,
                value_vnd=100.0,
                source="legacy",
                quality_flags={},
            )
        )
        s.commit()

        up = compute_daily_flow_features(s)
        assert up == 1
        rows = s.exec(select(DailyFlowFeature)).all()
        assert len(rows) == 1
        # day net flow is sum across rows
        assert rows[0].net_foreign_val_day == 23.0
