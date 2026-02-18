from __future__ import annotations

import datetime as dt
import math

import sqlalchemy as sa
from core.db.models import DailyIntradayFeature, FeatureLastProcessed, PriceOHLCV
from sqlmodel import Session, select

FEATURE_NAME = "daily_intraday_features"


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


def compute_daily_intraday_features(session: Session) -> int:
    last_date = _get_last_date(session)
    day_expr = sa.func.date(PriceOHLCV.timestamp)
    day_stmt = (
        select(PriceOHLCV.symbol, PriceOHLCV.source, day_expr.label("date_str"))
        .where(PriceOHLCV.timeframe == "1m")
        .group_by(PriceOHLCV.symbol, PriceOHLCV.source, day_expr)
        .order_by(PriceOHLCV.symbol, day_expr)
    )
    if last_date:
        day_stmt = day_stmt.where(
            PriceOHLCV.timestamp
            >= dt.datetime.combine(last_date + dt.timedelta(days=1), dt.time.min)
        )
    rows = session.exec(day_stmt).all()
    if not rows:
        return 0

    symbol_days = [
        (str(symbol), dt.date.fromisoformat(str(date_str)), str(source))
        for symbol, source, date_str in rows
    ]

    upserts = 0
    max_date: dt.date | None = None
    for symbol, day, source in symbol_days:
        bars = session.exec(
            select(PriceOHLCV)
            .where(PriceOHLCV.timeframe == "1m")
            .where(PriceOHLCV.symbol == symbol)
            .where(PriceOHLCV.source == source)
            .where(PriceOHLCV.timestamp >= dt.datetime.combine(day, dt.time.min))
            .where(
                PriceOHLCV.timestamp < dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min)
            )
            .order_by(PriceOHLCV.timestamp)
        ).all()
        if not bars:
            continue

        rv_sum = 0.0
        prev_close = None
        vol_day = 0.0
        vol_first_hour = 0.0
        for b in bars:
            close = float(b.close or 0.0)
            vol = float(b.volume or 0.0)
            vol_day += vol
            t = b.timestamp.time()
            if dt.time(9, 15) <= t <= dt.time(10, 15):
                vol_first_hour += vol
            if prev_close and prev_close > 0 and close > 0:
                lr = math.log(close / prev_close)
                rv_sum += lr * lr
            prev_close = close

        rv_day = math.sqrt(rv_sum)
        ratio = (vol_first_hour / vol_day) if vol_day > 0 else 0.0

        row = session.exec(
            select(DailyIntradayFeature)
            .where(DailyIntradayFeature.symbol == symbol)
            .where(DailyIntradayFeature.date == day)
            .where(DailyIntradayFeature.source == source)
        ).first()
        if row:
            row.rv_day = rv_day
            row.vol_first_hour_ratio = ratio
            session.add(row)
        else:
            session.add(
                DailyIntradayFeature(
                    symbol=symbol,
                    date=day,
                    source=source,
                    rv_day=rv_day,
                    vol_first_hour_ratio=ratio,
                )
            )
        upserts += 1
        if max_date is None or day > max_date:
            max_date = day

    if max_date is not None:
        _set_last_date(session, max_date)
    session.commit()
    return upserts
