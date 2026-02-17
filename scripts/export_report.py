from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from core.db.models import PriceOHLCV, Trade
from core.db.session import create_db_and_tables, get_engine
from core.fees_taxes import FeesTaxes
from core.portfolio.analytics import compute_positions_avg_cost
from core.settings import get_settings
from sqlmodel import Session, select

settings = get_settings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export portfolio report (CSV/HTML)")
    parser.add_argument("--portfolio-id", type=int, required=True)
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    broker = settings.BROKER_NAME

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        trades = session.exec(select(Trade).where(Trade.portfolio_id == args.portfolio_id)).all()
        if not trades:
            raise SystemExit("No trades found.")
        tdf = pd.DataFrame([t.model_dump() for t in trades])

        prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
        pdf = pd.DataFrame([p.model_dump() for p in prices])
        pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date
        latest = pdf.sort_values(["symbol", "date"]).groupby("symbol")["close"].last().to_dict()

        positions, realized = compute_positions_avg_cost(tdf, latest, fees, broker_name=broker)

        pos_df = pd.DataFrame([p.__dict__ for p in positions.values()])
        pos_csv = outdir / f"portfolio_{args.portfolio_id}_positions.csv"
        pos_df.to_csv(pos_csv, index=False)

        realized_csv = outdir / f"portfolio_{args.portfolio_id}_realized.csv"
        realized.to_csv(realized_csv, index=False)

        html = outdir / f"portfolio_{args.portfolio_id}_report.html"
        html.write_text(
            "<h1>Portfolio Report (MVP)</h1>"
            + "<h2>Positions</h2>"
            + pos_df.to_html(index=False)
            + "<h2>Realized</h2>"
            + realized.to_html(index=False),
            encoding="utf-8",
        )

        print(f"Exported: {pos_csv}, {realized_csv}, {html}")


if __name__ == "__main__":
    main()
