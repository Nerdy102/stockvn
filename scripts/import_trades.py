from __future__ import annotations

import argparse
import hashlib

import pandas as pd
from core.db.models import Portfolio, Trade
from core.db.session import create_db_and_tables, get_engine
from core.fees_taxes import FeesTaxes
from core.settings import get_settings
from sqlmodel import Session, select

settings = get_settings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import trades CSV into a portfolio")
    parser.add_argument("--csv", required=True, help="Path to trades CSV")
    parser.add_argument("--portfolio", default="Demo Portfolio", help="Portfolio name")
    args = parser.parse_args()

    create_db_and_tables(settings.DATABASE_URL)
    engine = get_engine(settings.DATABASE_URL)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    broker = settings.BROKER_NAME

    df = pd.read_csv(args.csv)
    with Session(engine) as session:
        p = session.exec(select(Portfolio).where(Portfolio.name == args.portfolio)).first()
        if p is None:
            p = Portfolio(name=args.portfolio)
            session.add(p)
            session.commit()
            session.refresh(p)

        inserted = 0
        for _, r in df.iterrows():
            raw = f"{p.id}|{r['trade_date']}|{r['symbol']}|{r['side']}|{r['quantity']}|{r['price']}|{r.get('strategy_tag','')}|{r.get('notes','')}"
            ext = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            if session.exec(select(Trade).where(Trade.external_id == ext)).first():
                continue

            qty = float(r["quantity"])
            px = float(r["price"])
            notional = qty * px
            commission = fees.commission(notional, broker)
            taxes = fees.sell_tax(notional) if str(r["side"]).upper() == "SELL" else 0.0

            t = Trade(
                portfolio_id=int(p.id or 0),
                trade_date=pd.to_datetime(r["trade_date"]).date(),
                symbol=str(r["symbol"]),
                side=str(r["side"]).upper(),
                quantity=qty,
                price=px,
                strategy_tag=str(r.get("strategy_tag", "")),
                notes=str(r.get("notes", "")),
                commission=commission,
                taxes=taxes,
                external_id=ext,
            )
            session.add(t)
            inserted += 1

        session.commit()
        print(f"Inserted {inserted} trades into portfolio #{p.id} ({p.name}).")


if __name__ == "__main__":
    main()
