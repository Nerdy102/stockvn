from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
from core.db.models import (
    AlphaPrediction,
    DailyFlowFeature,
    DailyIntradayFeature,
    DailyOrderbookFeature,
    ParquetManifest,
    PriceOHLCV,
    QuoteL2,
    TradeTape,
)
from core.settings import Settings
from sqlalchemy import func
from sqlmodel import Session, select


_DATASETS = [
    ("prices_ohlcv", PriceOHLCV, "timestamp"),
    ("quotes_l2", QuoteL2, "timestamp"),
    ("trades_tape", TradeTape, "timestamp"),
    ("daily_flow_features", DailyFlowFeature, "date"),
    ("daily_orderbook_features", DailyOrderbookFeature, "date"),
    ("daily_intraday_features", DailyIntradayFeature, "date"),
    ("alpha_predictions", AlphaPrediction, "as_of_date"),
]


def _schema_hash(df: pd.DataFrame) -> str:
    payload = "|".join(f"{c}:{str(t)}" for c, t in zip(df.columns, df.dtypes, strict=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_partitioned_parquet_for_day(
    session: Session,
    settings: Settings,
    as_of_date: dt.date,
) -> dict[str, int]:
    root = Path(settings.PARQUET_LAKE_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    out_counts: dict[str, int] = {}
    y, m, d = as_of_date.year, as_of_date.month, as_of_date.day

    for dataset, model, date_col in _DATASETS:
        col = getattr(model, date_col)
        q = select(model)
        if date_col in {"timestamp", "ts_utc"}:
            q = q.where(func.date(col) == as_of_date)
        else:
            q = q.where(col == as_of_date)
        rows = session.exec(q).all()
        if not rows:
            out_counts[dataset] = 0
            continue

        df = pd.DataFrame([r.model_dump() for r in rows])
        for c in df.columns:
            if df[c].dtype == "object":
                sample = next((v for v in df[c].tolist() if v is not None), None)
                if isinstance(sample, (dict, list)):
                    df[c] = df[c].map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
        dataset_dir = root / dataset / f"year={y}" / f"month={m:02d}" / f"day={d:02d}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_file = dataset_dir / "part-000.parquet"
        df.to_parquet(out_file, index=False, compression="zstd")

        shash = _schema_hash(df)
        row_count = int(len(df))
        mf = session.exec(
            select(ParquetManifest)
            .where(ParquetManifest.dataset == dataset)
            .where(ParquetManifest.year == y)
            .where(ParquetManifest.month == m)
            .where(ParquetManifest.day == d)
        ).first()
        if mf is None:
            mf = ParquetManifest(
                dataset=dataset,
                year=y,
                month=m,
                day=d,
                file_path=str(out_file),
                row_count=row_count,
                schema_hash=shash,
            )
        else:
            mf.file_path = str(out_file)
            mf.row_count = row_count
            mf.schema_hash = shash
        session.add(mf)
        out_counts[dataset] = row_count

    session.commit()
    return out_counts
