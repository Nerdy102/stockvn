from __future__ import annotations

import argparse
import datetime as dt

from core.db.session import create_db_and_tables, get_database_url, get_engine
from sqlmodel import Session
from worker_scheduler.jobs import predict_alpha_v3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Prediction date in YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    as_of_date = dt.date.fromisoformat(args.date)
    db_url = get_database_url()
    create_db_and_tables(db_url)
    engine = get_engine(db_url)
    with Session(engine) as session:
        n = predict_alpha_v3(session, as_of_date=as_of_date)
    print({"predictions": n, "as_of_date": args.date})


if __name__ == "__main__":
    main()
