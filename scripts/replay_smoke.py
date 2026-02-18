from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sqlmodel import Session, select

from core.calendar_vn import get_trading_calendar_vn
from core.db.models import EventLog
from core.db.session import create_db_and_tables, get_engine
from core.fees_taxes import FeesTaxes
from core.market_rules import load_market_rules
from core.replay.pipeline import UnifiedTradingPipeline
from scripts.replay_events import replay_into_redis
from tests.helpers_redis_fake import FakeRedisCompat

FIXTURE = Path("tests/fixtures/replay/event_log_fixture.jsonl")


def _cost_model(_: float) -> float:
    return 0.0005


def _no_ca(_: dict, __: dict) -> None:
    return None


def _load_fixture(session: Session) -> None:
    session.exec(EventLog.__table__.delete())
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        session.add(
            EventLog(
                ts_utc=dt.datetime.fromisoformat(row["ts_utc"]),
                source=row["source"],
                event_type=row["event_type"],
                symbol=row.get("symbol"),
                payload_json=row["payload_json"],
                payload_hash=row["payload_hash"],
                run_id="fixture-2d",
            )
        )
    session.commit()


def _pipeline() -> UnifiedTradingPipeline:
    return UnifiedTradingPipeline(
        market_rules=load_market_rules("configs/market_rules_vn.yaml"),
        fees_taxes=FeesTaxes(0.001, 0.05, 0.0015, {}),
        cost_model=_cost_model,
        corporate_action_ledger=_no_ca,
        calendar_vn=get_trading_calendar_vn(),
    )


def run_smoke() -> None:
    nav = 100_000_000.0
    db_url = "sqlite:///./vn_invest.db"
    create_db_and_tables(db_url)
    engine = get_engine(db_url)

    with Session(engine) as session:
        _load_fixture(session)
        events = list(
            session.exec(
                select(EventLog).where(EventLog.run_id == "fixture-2d").order_by(EventLog.ts_utc, EventLog.id)
            )
        )

    canonical_events = [
        {
            "id": e.id,
            "ts_utc": e.ts_utc,
            "event_type": e.event_type,
            "symbol": e.symbol,
            "payload_json": e.payload_json,
        }
        for e in events
    ]

    backtest = _pipeline().run(canonical_events, initial_cash=nav)

    redis_client = FakeRedisCompat()
    replay_into_redis(events, redis_client, speed="max")

    paper_events: list[dict] = []
    for stream in ["ssi:signal", "ssi:bar"]:
        for _, fields in redis_client.xrange(stream):
            paper_events.append(
                {
                    "ts_utc": fields["ts_utc"],
                    "event_type": fields["event_type"],
                    "symbol": fields.get("symbol") or None,
                    "payload_json": json.loads(fields["payload"]),
                }
            )

    paper = _pipeline().run(paper_events, initial_cash=nav)

    if backtest.positions != paper.positions:
        raise AssertionError(f"positions EOD mismatch: {backtest.positions} != {paper.positions}")
    if backtest.position_eod != paper.position_eod:
        raise AssertionError(f"position ledger mismatch: {backtest.position_eod} != {paper.position_eod}")

    if backtest.cash_ledger.keys() != paper.cash_ledger.keys():
        raise AssertionError("cash ledger dates mismatch")
    for d in backtest.cash_ledger:
        if abs(backtest.cash_ledger[d] - paper.cash_ledger[d]) > 1.0:
            raise AssertionError(
                f"cash ledger mismatch on {d}: {backtest.cash_ledger[d]} != {paper.cash_ledger[d]}"
            )
    if abs(backtest.cash - paper.cash) > 1.0:
        raise AssertionError(f"cash mismatch: {backtest.cash} != {paper.cash}")
    if abs(backtest.fees_total - paper.fees_total) > 1.0:
        raise AssertionError("fees mismatch")
    if abs(backtest.taxes_total - paper.taxes_total) > 1.0:
        raise AssertionError("taxes mismatch")
    pnl_tol = 1e-6 * nav
    if abs(backtest.realized_pnl - paper.realized_pnl) > pnl_tol:
        raise AssertionError("realized pnl mismatch")
    print(
        json.dumps(
            {
                "status": "ok",
                "positions": backtest.positions,
                "cash": backtest.cash,
                "cash_ledger": backtest.cash_ledger,
            }
        )
    )


if __name__ == "__main__":
    run_smoke()
