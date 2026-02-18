import datetime as dt

import pandas as pd
from sqlmodel import Session, SQLModel, create_engine

from core.db.models import CorporateAction
from worker_scheduler.jobs import _adjust_daily_prices_with_ca


def test_worker_adjusts_daily_prices_by_default_for_ca() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(
            CorporateAction(
                symbol="AAA",
                action_type="SPLIT",
                ex_date=dt.date(2025, 1, 3),
                params_json={"split_factor": 2.0},
                source="test",
                raw_json={},
            )
        )
        s.commit()

        bars = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "AAA"],
                "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                "open": [100.0, 102.0, 51.0],
                "high": [100.0, 102.0, 51.0],
                "low": [100.0, 102.0, 51.0],
                "close": [100.0, 102.0, 51.0],
                "volume": [1000.0, 1000.0, 4000.0],
                "timeframe": ["1D", "1D", "1D"],
                "value_vnd": [100.0, 100.0, 100.0],
                "source": ["x", "x", "x"],
                "quality_flags": [{}, {}, {}],
            }
        )

        out = _adjust_daily_prices_with_ca(s, bars)

        # Pre-ex rows are backward adjusted and no date is removed.
        assert len(out) == 3
        close_d2 = float(out.loc[out["timestamp"] == pd.Timestamp("2025-01-02"), "close"].iloc[0])
        vol_d2 = float(out.loc[out["timestamp"] == pd.Timestamp("2025-01-02"), "volume"].iloc[0])
        assert close_d2 == 51.0
        assert vol_d2 == 2000.0
