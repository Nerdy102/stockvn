from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from core.data_lake.feature_source import load_table_df
from core.data_lake.parquet_export import export_partitioned_parquet_for_day
from core.db.models import AlphaPrediction, ParquetManifest, PriceOHLCV, QuoteL2, Ticker
from core.db.session import get_engine
from core.settings import Settings
from sqlmodel import SQLModel, Session, select


def _seed(session: Session, day: dt.date) -> None:
    session.add(Ticker(symbol="AAA", name="AAA", exchange="HOSE", sector="Tech", industry="Soft"))
    session.add(
        PriceOHLCV(
            symbol="AAA",
            timeframe="1D",
            timestamp=dt.datetime.combine(day, dt.time(0, 0)),
            open=10,
            high=11,
            low=9,
            close=10.5,
            volume=1000,
            value_vnd=10_500,
        )
    )
    session.add(
        QuoteL2(
            symbol="AAA",
            timestamp=dt.datetime.combine(day, dt.time(1, 0)),
            bids={"p": [10.0], "v": [100.0]},
            asks={"p": [11.0], "v": [120.0]},
            source="fixture",
        )
    )
    session.add(
        AlphaPrediction(
            model_id="alpha_v3",
            symbol="AAA",
            as_of_date=day,
            score=0.12,
            mu=0.11,
            uncert=0.03,
            pred_base=0.10,
        )
    )
    session.commit()


def test_parquet_export_creates_partitions_and_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    day = dt.date(2025, 2, 14)

    settings = Settings(
        DATABASE_URL=f"sqlite:///{db_path}",
        PARQUET_LAKE_ROOT=str(tmp_path / "lake"),
        ENABLE_DUCKDB_FAST_PATH=False,
    )

    with Session(engine) as session:
        _seed(session, day)
        out = export_partitioned_parquet_for_day(session, settings=settings, as_of_date=day)
        assert out["prices_ohlcv"] == 1
        assert out["quotes_l2"] == 1
        assert out["alpha_predictions"] == 1

        mf = session.exec(
            select(ParquetManifest)
            .where(ParquetManifest.dataset == "prices_ohlcv")
            .where(ParquetManifest.year == 2025)
            .where(ParquetManifest.month == 2)
            .where(ParquetManifest.day == 14)
        ).first()
        assert mf is not None
        assert mf.row_count == 1
        assert len(mf.schema_hash) == 64

    partition = tmp_path / "lake" / "prices_ohlcv" / "year=2025" / "month=02" / "day=14" / "part-000.parquet"
    assert partition.exists()


@pytest.mark.skipif(__import__("importlib").util.find_spec("duckdb") is None, reason="duckdb not installed")
def test_duckdb_fast_path_equals_db_path(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    day = dt.date(2025, 2, 14)

    base_settings = Settings(DATABASE_URL=f"sqlite:///{db_path}", PARQUET_LAKE_ROOT=str(tmp_path / "lake"))

    with Session(engine) as session:
        _seed(session, day)
        export_partitioned_parquet_for_day(session, settings=base_settings, as_of_date=day)

        db_df = load_table_df(
            session,
            model=PriceOHLCV,
            table_name="prices_ohlcv",
            date_col="timestamp",
            settings=Settings(DATABASE_URL=f"sqlite:///{db_path}", PARQUET_LAKE_ROOT=str(tmp_path / "lake"), ENABLE_DUCKDB_FAST_PATH=False),
            start_date=day,
            end_date=day,
            symbols=["AAA"],
        )
        dd_df = load_table_df(
            session,
            model=PriceOHLCV,
            table_name="prices_ohlcv",
            date_col="timestamp",
            settings=Settings(DATABASE_URL=f"sqlite:///{db_path}", PARQUET_LAKE_ROOT=str(tmp_path / "lake"), ENABLE_DUCKDB_FAST_PATH=True),
            start_date=day,
            end_date=day,
            symbols=["AAA"],
        )

    assert not db_df.empty and not dd_df.empty
    cols = sorted(set(db_df.columns).intersection(dd_df.columns))
    pd.testing.assert_frame_equal(
        db_df[cols].sort_values(cols).reset_index(drop=True),
        dd_df[cols].sort_values(cols).reset_index(drop=True),
        rtol=1e-9,
        atol=1e-9,
        check_dtype=False,
    )
