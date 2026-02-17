from __future__ import annotations

import datetime as dt

from core.db.models import FactorScore, Fundamental, IndicatorValue, PriceOHLCV, Ticker
from sqlmodel import Session, SQLModel, create_engine, select
from worker_scheduler.jobs import compute_factor_scores, compute_indicators


def test_compute_jobs_idempotent() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(
            Ticker(
                symbol="AAA",
                name="AAA",
                exchange="HOSE",
                sector="Tech",
                industry="IT",
                shares_outstanding=1_000_000,
            )
        )
        s.add(
            Fundamental(
                symbol="AAA",
                as_of_date=dt.date(2025, 1, 1),
                public_date=dt.date(2025, 2, 15),
                sector="Tech",
                revenue_ttm_vnd=100,
                net_income_ttm_vnd=10,
                gross_profit_ttm_vnd=20,
                ebitda_ttm_vnd=15,
                cfo_ttm_vnd=12,
                dividends_ttm_vnd=1,
                total_assets_vnd=200,
                total_liabilities_vnd=80,
                equity_vnd=120,
                net_debt_vnd=10,
            )
        )
        s.add(
            PriceOHLCV(
                symbol="AAA",
                timeframe="1D",
                timestamp=dt.datetime(2025, 3, 1),
                open=10,
                high=11,
                low=9,
                close=10.5,
                volume=1000,
                value_vnd=10500,
            )
        )
        s.add(
            PriceOHLCV(
                symbol="VNINDEX",
                timeframe="1D",
                timestamp=dt.datetime(2025, 3, 1),
                open=1000,
                high=1010,
                low=990,
                close=1005,
                volume=1_000_000,
                value_vnd=1_005_000_000,
            )
        )
        s.commit()

        compute_indicators(s)
        compute_factor_scores(s)
        ind_1 = len(s.exec(select(IndicatorValue)).all())
        fac_1 = len(s.exec(select(FactorScore)).all())

        compute_indicators(s)
        compute_factor_scores(s)
        ind_2 = len(s.exec(select(IndicatorValue)).all())
        fac_2 = len(s.exec(select(FactorScore)).all())

    assert ind_2 == ind_1
    assert fac_2 == fac_1
