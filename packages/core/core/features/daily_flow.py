from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import sqlalchemy as sa
from core.db.models import DailyFlowFeature, FeatureLastProcessed, MarketDailyMeta, PriceOHLCV
from sqlmodel import Session, select

FEATURE_NAME = "daily_flow_features"


def _get_last_date(session: Session, symbol: str = "") -> dt.date | None:
    row = session.exec(
        select(FeatureLastProcessed)
        .where(FeatureLastProcessed.feature_name == FEATURE_NAME)
        .where(FeatureLastProcessed.symbol == symbol)
    ).first()
    return row.last_date if row else None


def _set_last_date(session: Session, last_date: dt.date, symbol: str = "") -> None:
    row = session.exec(
        select(FeatureLastProcessed)
        .where(FeatureLastProcessed.feature_name == FEATURE_NAME)
        .where(FeatureLastProcessed.symbol == symbol)
    ).first()
    if row:
        row.last_date = last_date
        row.updated_at = dt.datetime.utcnow()
        session.add(row)
    else:
        session.add(
            FeatureLastProcessed(
                feature_name=FEATURE_NAME,
                symbol=symbol,
                last_date=last_date,
            )
        )


def _load_flow_daily_aggregates(
    session: Session, lookback_start: dt.date | None
) -> list[tuple[str, str, str, float, float | None, float | None]]:
    m = MarketDailyMeta.__table__
    day_expr = sa.func.date(m.c.timestamp)

    where_clause = sa.true()
    if lookback_start:
        where_clause = m.c.timestamp >= dt.datetime.combine(lookback_start, dt.time.min)

    sum_stmt = (
        sa.select(
            m.c.symbol.label("symbol"),
            m.c.source.label("source"),
            day_expr.label("date_str"),
            (
                sa.func.coalesce(sa.func.sum(m.c.foreign_buy_value), 0.0)
                - sa.func.coalesce(sa.func.sum(m.c.foreign_sell_value), 0.0)
            ).label("net_foreign_val_day"),
        )
        .where(where_clause)
        .group_by(m.c.symbol, m.c.source, day_expr)
        .subquery()
    )

    latest_ranked = (
        sa.select(
            m.c.symbol.label("symbol"),
            m.c.source.label("source"),
            day_expr.label("date_str"),
            m.c.current_room.label("current_room"),
            m.c.total_room.label("total_room"),
            sa.func.row_number()
            .over(
                partition_by=(m.c.symbol, m.c.source, day_expr),
                order_by=(m.c.timestamp.desc(), m.c.id.desc()),
            )
            .label("rn"),
        )
        .where(where_clause)
        .subquery()
    )

    latest_stmt = (
        sa.select(
            latest_ranked.c.symbol,
            latest_ranked.c.source,
            latest_ranked.c.date_str,
            latest_ranked.c.current_room,
            latest_ranked.c.total_room,
        )
        .where(latest_ranked.c.rn == 1)
        .subquery()
    )

    stmt = (
        sa.select(
            sum_stmt.c.symbol,
            sum_stmt.c.source,
            sum_stmt.c.date_str,
            sum_stmt.c.net_foreign_val_day,
            latest_stmt.c.current_room,
            latest_stmt.c.total_room,
        )
        .join(
            latest_stmt,
            (sum_stmt.c.symbol == latest_stmt.c.symbol)
            & (sum_stmt.c.source == latest_stmt.c.source)
            & (sum_stmt.c.date_str == latest_stmt.c.date_str),
        )
        .order_by(sum_stmt.c.symbol, sum_stmt.c.source, sum_stmt.c.date_str)
    )
    return list(session.exec(stmt).all())


def compute_daily_flow_features(session: Session) -> int:
    last_date = _get_last_date(session)
    start_date = (last_date + dt.timedelta(days=1)) if last_date else None
    lookback_start = (start_date - dt.timedelta(days=25)) if start_date else None

    flow_rows = _load_flow_daily_aggregates(session, lookback_start)
    if not flow_rows:
        return 0

    daily = pd.DataFrame(
        [
            {
                "symbol": str(symbol),
                "source": str(source),
                "date": dt.date.fromisoformat(str(date_str)),
                "net_foreign_val_day": float(net_val or 0.0),
                "current_room": current_room,
                "total_room": total_room,
            }
            for symbol, source, date_str, net_val, current_room, total_room in flow_rows
        ]
    ).sort_values(["symbol", "source", "date"])

    daily["net_foreign_val_5d"] = (
        daily.groupby(["symbol", "source"])["net_foreign_val_day"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    daily["net_foreign_val_20d"] = (
        daily.groupby(["symbol", "source"])["net_foreign_val_day"]
        .rolling(20, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    adv_stmt = select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")
    if lookback_start:
        adv_stmt = adv_stmt.where(
            PriceOHLCV.timestamp >= dt.datetime.combine(lookback_start, dt.time.min)
        )
    adv_rows = session.exec(adv_stmt).all()
    if adv_rows:
        adf = pd.DataFrame([r.model_dump() for r in adv_rows])
        adf["date"] = pd.to_datetime(adf["timestamp"]).dt.date
        adf = adf.sort_values(["symbol", "date"])
        adf["adv20_value"] = (
            adf.groupby("symbol")["value_vnd"]
            .rolling(20, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        daily = daily.merge(
            adf[["symbol", "date", "adv20_value"]], on=["symbol", "date"], how="left"
        )
    else:
        daily["adv20_value"] = np.nan

    daily["foreign_flow_intensity"] = daily["net_foreign_val_20d"] / daily["adv20_value"].replace(
        0, np.nan
    )
    daily["foreign_flow_intensity"] = daily["foreign_flow_intensity"].fillna(0.0)

    current_room = pd.to_numeric(daily["current_room"], errors="coerce")
    total_room = pd.to_numeric(daily["total_room"], errors="coerce").replace(0, np.nan)
    util = 1.0 - (current_room / total_room)
    daily["foreign_room_util"] = pd.to_numeric(util, errors="coerce")

    if start_date:
        daily = daily[daily["date"] >= start_date]
    if daily.empty:
        return 0

    upserts = 0
    for _, r in daily.iterrows():
        row = session.exec(
            select(DailyFlowFeature)
            .where(DailyFlowFeature.symbol == str(r["symbol"]))
            .where(DailyFlowFeature.date == r["date"])
            .where(DailyFlowFeature.source == str(r["source"]))
        ).first()
        payload = dict(
            net_foreign_val_day=float(r.get("net_foreign_val_day") or 0.0),
            net_foreign_val_5d=float(r.get("net_foreign_val_5d") or 0.0),
            net_foreign_val_20d=float(r.get("net_foreign_val_20d") or 0.0),
            foreign_flow_intensity=float(r.get("foreign_flow_intensity") or 0.0),
            foreign_room_util=(
                None if pd.isna(r.get("foreign_room_util")) else float(r.get("foreign_room_util"))
            ),
        )
        if row:
            for k, v in payload.items():
                setattr(row, k, v)
            session.add(row)
        else:
            session.add(
                DailyFlowFeature(
                    symbol=str(r["symbol"]),
                    date=r["date"],
                    source=str(r["source"]),
                    **payload,
                )
            )
        upserts += 1

    _set_last_date(session, max(daily["date"]))
    session.commit()
    return upserts
