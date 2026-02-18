from __future__ import annotations

import datetime as dt

from core.calendar_vn import get_trading_calendar_vn
from core.db.models import IndexMembership, PriceOHLCV, TickerLifecycle
from core.db.session import get_engine
from core.universe.manager import UniverseManager
from sqlmodel import Session, SQLModel


def _seed_symbol(
    session: Session,
    symbol: str,
    start_date: dt.date,
    n_days: int,
    value_vnd: float,
    volume: float = 1_000.0,
) -> list[dt.date]:
    cal = get_trading_calendar_vn()
    days: list[dt.date] = []
    cur = start_date
    while len(days) < n_days:
        if cal.is_trading_day(cur):
            days.append(cur)
        cur += dt.timedelta(days=1)

    for d in days:
        session.add(
            PriceOHLCV(
                symbol=symbol,
                timeframe="1D",
                timestamp=dt.datetime.combine(d, dt.time(0, 0)),
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.0,
                volume=volume,
                value_vnd=value_vnd,
                source="test",
                quality_flags={},
            )
        )
    return days


def test_delist_after_last_date_excluded(tmp_path) -> None:
    db = f"sqlite:///{tmp_path}/u1.db"
    engine = get_engine(db)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        days = _seed_symbol(s, "AAA", dt.date(2023, 1, 2), n_days=420, value_vnd=2_000_000_000)
        s.add(
            TickerLifecycle(
                symbol="AAA",
                first_trading_date=days[0],
                last_trading_date=days[-2],
                exchange="HOSE",
                sectype="stock",
                sector="TECH",
                source="test",
            )
        )
        s.commit()

        symbols, breakdown = UniverseManager(s).universe(days[-1], "ALL")
        assert "AAA" not in symbols
        assert breakdown == {"inactive_lifecycle": 1}


def test_membership_intervals_no_lookahead(tmp_path) -> None:
    db = f"sqlite:///{tmp_path}/u2.db"
    engine = get_engine(db)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        days = _seed_symbol(s, "AAA", dt.date(2023, 1, 2), n_days=420, value_vnd=2_000_000_000)
        s.add(
            TickerLifecycle(
                symbol="AAA",
                first_trading_date=days[0],
                last_trading_date=None,
                exchange="HOSE",
                sectype="stock",
                sector="TECH",
                source="test",
            )
        )
        s.add(
            IndexMembership(
                index_code="VN30",
                symbol="AAA",
                start_date=days[300],
                end_date=days[350],
                source="test",
            )
        )
        s.commit()

        mgr = UniverseManager(s)
        before, _ = mgr.universe(days[299], "VN30")
        inside, _ = mgr.universe(days[320], "VN30")
        after, _ = mgr.universe(days[351], "VN30")

        assert "AAA" not in before
        assert "AAA" in inside
        assert "AAA" not in after


def test_exclusion_breakdown_stable_and_adv20_is_pit(tmp_path) -> None:
    db = f"sqlite:///{tmp_path}/u3.db"
    engine = get_engine(db)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        days_a = _seed_symbol(s, "AAA", dt.date(2023, 1, 2), n_days=320, value_vnd=2_000_000_000)
        days_b = _seed_symbol(s, "BBB", dt.date(2023, 1, 2), n_days=320, value_vnd=200_000_000)
        _seed_symbol(s, "CCC", dt.date(2024, 1, 2), n_days=100, value_vnd=2_000_000_000)

        eval_date = get_trading_calendar_vn().next_trading_day(days_b[-1], 1)

        # PIT check: boost on-date liquidity for BBB; should still be excluded from past-20 ADV.
        s.add(
            PriceOHLCV(
                symbol="BBB",
                timeframe="1D",
                timestamp=dt.datetime.combine(eval_date, dt.time(0, 0)),
                open=10,
                high=11,
                low=9,
                close=10,
                volume=1_000,
                value_vnd=20_000_000_000,
                source="test",
                quality_flags={},
            )
        )

        for sym, first, last, sectype in [
            ("AAA", days_a[0], None, "stock"),
            ("BBB", days_b[0], None, "stock"),
            ("CCC", dt.date(2024, 1, 2), None, "stock"),
            ("ETF1", days_a[0], None, "etf"),
        ]:
            s.add(
                TickerLifecycle(
                    symbol=sym,
                    first_trading_date=first,
                    last_trading_date=last,
                    exchange="HOSE",
                    sectype=sectype,
                    sector="GEN",
                    source="test",
                )
            )

        # No-trade day for AAA on evaluation day
        s.add(
            PriceOHLCV(
                symbol="AAA",
                timeframe="1D",
                timestamp=dt.datetime.combine(eval_date, dt.time(0, 0)),
                open=10,
                high=10,
                low=10,
                close=10,
                volume=0,
                value_vnd=0,
                source="test",
                quality_flags={},
            )
        )

        s.commit()

        symbols, breakdown = UniverseManager(s).universe(eval_date, "ALL")

        assert symbols == []
        assert breakdown == {
            "adv20_below_threshold": 1,
            "insufficient_history_260": 1,
            "no_trade_day": 1,
            "sectype_not_stock": 1,
        }
