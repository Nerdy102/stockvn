from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from scripts import fetch_eodhd_vn_universe10 as fetch_script
from scripts.seed_demo_data import main as seed_demo_data_main


def _safe_delete_sqlite_db(database_url: str) -> bool:
    prefix = "sqlite:///./"
    if not database_url.startswith(prefix):
        print(
            f"[WARN] --reset-db skipped: only supports urls like {prefix}... (got: {database_url})"
        )
        return False
    db_name = database_url[len(prefix) :]
    db_path = Path.cwd() / db_name
    if db_path.suffix not in {".db", ".sqlite"}:
        print(f"[WARN] --reset-db skipped: refusing non-db file {db_path}")
        return False
    if db_path.exists():
        db_path.unlink()
        print(f"[OK] Removed database file: {db_path}")
        return True
    print(f"[INFO] Database file not found, nothing to remove: {db_path}")
    return False


def _print_data_checklist(prices_path: Path) -> None:
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    n_rows = len(prices)
    n_symbols = int(prices["symbol"].nunique())
    duplicates = int(prices.duplicated(subset=["symbol", "date"]).sum())
    missing_volume = float(prices["volume"].isna().mean() * 100)
    missing_close = float(prices["close"].isna().mean() * 100)

    print("\n=== VN10 bootstrap checklist ===")
    print(f"Rows: {n_rows}")
    print(f"Symbols: {n_symbols}")
    print(f"Duplicate (symbol,date): {duplicates}")
    print(f"Missing volume: {missing_volume:.2f}%")
    print(f"Missing close: {missing_close:.2f}%")

    ranges = prices.groupby("symbol")["date"].agg(["min", "max", "count"]).sort_index()
    print("\nPer-symbol date range:")
    print(ranges.to_string())

    low_days = ranges[ranges["count"] < 150]
    if not low_days.empty:
        print("\n[WARN] Symbols with too few days (<150):")
        print(low_days[["count"]].to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch VN10 EODHD data, optional DB reset, then seed demo DB"
    )
    parser.add_argument("--from", dest="from_date", default=None)
    parser.add_argument("--to", dest="to_date", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-prices", default="data_demo/prices_demo_1d.csv")
    parser.add_argument("--out-tickers", default="data_demo/tickers_demo.csv")
    parser.add_argument("--reset-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_from, default_to = fetch_script._default_from_to()

    fetch_script.run(
        from_date=args.from_date or os.getenv("FROM", default_from),
        to_date=args.to_date or os.getenv("TO", default_to),
        refresh=bool(args.refresh),
        out_prices=Path(args.out_prices),
        out_tickers=Path(args.out_tickers),
    )

    if args.reset_db:
        _safe_delete_sqlite_db(os.getenv("DATABASE_URL", "sqlite:///./vn_invest.db"))

    seed_demo_data_main()
    _print_data_checklist(Path(args.out_prices))


if __name__ == "__main__":
    main()
