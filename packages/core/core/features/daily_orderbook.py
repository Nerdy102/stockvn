from __future__ import annotations

import datetime as dt

import sqlalchemy as sa
from core.db.models import DailyOrderbookFeature, FeatureLastProcessed, QuoteL2
from sqlmodel import Session, select

FEATURE_NAME = "daily_orderbook_features"


def _get_last_date(session: Session) -> dt.date | None:
    row = session.exec(
        select(FeatureLastProcessed)
        .where(FeatureLastProcessed.feature_name == FEATURE_NAME)
        .where(FeatureLastProcessed.symbol == "")
    ).first()
    return row.last_date if row else None


def _set_last_date(session: Session, last_date: dt.date) -> None:
    row = session.exec(
        select(FeatureLastProcessed)
        .where(FeatureLastProcessed.feature_name == FEATURE_NAME)
        .where(FeatureLastProcessed.symbol == "")
    ).first()
    if row:
        row.last_date = last_date
        row.updated_at = dt.datetime.utcnow()
        session.add(row)
    else:
        session.add(FeatureLastProcessed(feature_name=FEATURE_NAME, symbol="", last_date=last_date))


def _json_num(expr: sa.ColumnElement, path: str, dialect: str) -> sa.ColumnElement:
    if dialect == "postgresql":
        clean = path.lstrip("$").strip(".")
        parts = clean.replace("]", "").replace("[", ".").split(".")
        current = expr
        for part in [p for p in parts if p]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = current[part]
        return sa.cast(current.astext, sa.Float)
    return sa.cast(sa.func.json_extract(expr, path), sa.Float)


def compute_daily_orderbook_features(session: Session) -> int:
    last_date = _get_last_date(session)
    q = QuoteL2.__table__
    dialect = session.get_bind().dialect.name

    bid_v1 = sa.func.coalesce(_json_num(q.c.bids, "$.volumes[0]", dialect), 0.0)
    ask_v1 = sa.func.coalesce(_json_num(q.c.asks, "$.volumes[0]", dialect), 0.0)
    bid_v2 = sa.func.coalesce(_json_num(q.c.bids, "$.volumes[1]", dialect), 0.0)
    ask_v2 = sa.func.coalesce(_json_num(q.c.asks, "$.volumes[1]", dialect), 0.0)
    bid_v3 = sa.func.coalesce(_json_num(q.c.bids, "$.volumes[2]", dialect), 0.0)
    ask_v3 = sa.func.coalesce(_json_num(q.c.asks, "$.volumes[2]", dialect), 0.0)

    bid_p1 = sa.func.coalesce(_json_num(q.c.bids, "$.prices[0]", dialect), 0.0)
    ask_p1 = sa.func.coalesce(_json_num(q.c.asks, "$.prices[0]", dialect), 0.0)

    date_expr = sa.func.date(q.c.ts_utc)
    imb1 = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-9)
    bid3 = bid_v1 + bid_v2 + bid_v3
    ask3 = ask_v1 + ask_v2 + ask_v3
    imb3 = (bid3 - ask3) / (bid3 + ask3 + 1e-9)
    mid = (ask_p1 + bid_p1) / 2.0
    spread = sa.case((mid > 0, (ask_p1 - bid_p1) / mid), else_=None)

    stmt = (
        sa.select(
            q.c.symbol,
            q.c.source,
            date_expr.label("date_str"),
            sa.func.avg(imb1).label("imb_1_day"),
            sa.func.avg(imb3).label("imb_3_day"),
            sa.func.avg(spread).label("spread_day"),
        )
        .group_by(q.c.symbol, q.c.source, date_expr)
        .order_by(q.c.symbol, date_expr)
    )
    if last_date:
        stmt = stmt.where(
            q.c.ts_utc >= dt.datetime.combine(last_date + dt.timedelta(days=1), dt.time.min)
        )

    rows = session.exec(stmt).all()
    if not rows:
        return 0

    upserts = 0
    max_date = None
    for r in rows:
        row_date = dt.date.fromisoformat(str(r.date_str))
        if max_date is None or row_date > max_date:
            max_date = row_date
        row = session.exec(
            select(DailyOrderbookFeature)
            .where(DailyOrderbookFeature.symbol == r.symbol)
            .where(DailyOrderbookFeature.date == row_date)
            .where(DailyOrderbookFeature.source == r.source)
        ).first()
        if row:
            row.imb_1_day = float(r.imb_1_day or 0.0)
            row.imb_3_day = float(r.imb_3_day or 0.0)
            row.spread_day = float(r.spread_day or 0.0)
            session.add(row)
        else:
            session.add(
                DailyOrderbookFeature(
                    symbol=r.symbol,
                    source=r.source,
                    date=row_date,
                    imb_1_day=float(r.imb_1_day or 0.0),
                    imb_3_day=float(r.imb_3_day or 0.0),
                    spread_day=float(r.spread_day or 0.0),
                )
            )
        upserts += 1

    _set_last_date(session, max_date)
    session.commit()
    return upserts
