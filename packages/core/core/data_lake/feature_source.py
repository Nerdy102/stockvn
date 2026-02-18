from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
from core.settings import Settings
from sqlmodel import Session, select


def load_table_df(
    session: Session,
    model: Any,
    table_name: str,
    date_col: str,
    settings: Settings,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if settings.ENABLE_DUCKDB_FAST_PATH:
        df = _load_with_duckdb(table_name, date_col, settings, start_date, end_date, symbols)
        if df is not None:
            return df
    return _load_from_db(session, model, date_col, start_date, end_date, symbols)


def _load_with_duckdb(
    table_name: str,
    date_col: str,
    settings: Settings,
    start_date: dt.date | None,
    end_date: dt.date | None,
    symbols: list[str] | None,
) -> pd.DataFrame | None:
    try:
        import duckdb
    except Exception:
        return None

    root = Path(settings.PARQUET_LAKE_ROOT) / table_name
    if not root.exists():
        return None

    con = duckdb.connect(database=settings.DUCKDB_PATH)
    try:
        glob = str(root / "year=*" / "month=*" / "day=*" / "*.parquet")
        sql = f"SELECT * FROM read_parquet('{glob}', hive_partitioning=1) WHERE 1=1"
        params: list[Any] = []
        if start_date is not None:
            sql += f" AND CAST({date_col} AS DATE) >= ?"
            params.append(start_date)
        if end_date is not None:
            sql += f" AND CAST({date_col} AS DATE) <= ?"
            params.append(end_date)
        if symbols:
            placeholders = ",".join(["?"] * len(symbols))
            sql += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        return con.execute(sql, params).df()
    except Exception:
        return None
    finally:
        con.close()


def _load_from_db(
    session: Session,
    model: Any,
    date_col: str,
    start_date: dt.date | None,
    end_date: dt.date | None,
    symbols: list[str] | None,
) -> pd.DataFrame:
    q = select(model)
    col = getattr(model, date_col)
    if start_date is not None:
        q = q.where(col >= start_date)
    if end_date is not None:
        q = q.where(col <= end_date)
    if symbols and hasattr(model, "symbol"):
        q = q.where(getattr(model, "symbol").in_(symbols))
    rows = session.exec(q).all()
    return pd.DataFrame([r.model_dump() for r in rows]) if rows else pd.DataFrame()
