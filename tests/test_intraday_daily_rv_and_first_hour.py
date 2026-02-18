from __future__ import annotations

import datetime as dt
import math

from sqlmodel import Session, SQLModel, create_engine, select

from core.db.models import DailyIntradayFeature, PriceOHLCV
from core.features.daily_intraday import compute_daily_intraday_features


def test_intraday_daily_rv_and_first_hour() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    bars = [
        (dt.datetime(2024, 1, 3, 9, 15), 100.0, 10.0),
        (dt.datetime(2024, 1, 3, 9, 16), 101.0, 20.0),
        (dt.datetime(2024, 1, 3, 10, 15), 100.0, 30.0),
        (dt.datetime(2024, 1, 3, 10, 16), 102.0, 40.0),
    ]

    with Session(engine) as s:
        for ts, close, vol in bars:
            s.add(
                PriceOHLCV(
                    symbol="AAA",
                    timeframe="1m",
                    timestamp=ts,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=vol,
                    value_vnd=0.0,
                    source="sim",
                    quality_flags={},
                )
            )
        s.commit()

        up = compute_daily_intraday_features(s)
        assert up == 1

        row = s.exec(select(DailyIntradayFeature)).first()
        assert row is not None

        rets = [math.log(101.0 / 100.0), math.log(100.0 / 101.0), math.log(102.0 / 100.0)]
        expected_rv = math.sqrt(sum(x * x for x in rets))
        assert abs(row.rv_day - expected_rv) < 1e-12
        assert abs(row.vol_first_hour_ratio - (60.0 / 100.0)) < 1e-12
