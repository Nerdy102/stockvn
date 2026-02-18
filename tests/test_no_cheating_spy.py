from __future__ import annotations

import datetime as dt
import json

from core.calendar_vn import TradingCalendarVN
from core.db.models import EventLog
from core.fees_taxes import FeesTaxes
from core.market_rules import load_market_rules
from core.replay.pipeline import UnifiedTradingPipeline
from scripts.replay_events import replay_into_redis
from tests.helpers_redis_fake import FakeRedisCompat


class Spy:
    def __init__(self) -> None:
        self.calls = {"market_rules": 0, "fees": 0, "tax": 0, "cost": 0, "ca": 0, "calendar": 0}


class SpyMarketRules:
    def __init__(self, base, spy: Spy) -> None:
        self._base = base
        self._spy = spy

    def round_price(self, price: float, direction: str = "nearest") -> float:
        self._spy.calls["market_rules"] += 1
        return self._base.round_price(price, direction=direction)


class SpyFees:
    def __init__(self, base: FeesTaxes, spy: Spy) -> None:
        self._base = base
        self._spy = spy

    def commission(self, notional: float, symbol: str | None = None) -> float:
        self._spy.calls["fees"] += 1
        return self._base.commission(notional, symbol)

    def sell_tax(self, notional: float, symbol: str | None = None) -> float:
        del symbol
        self._spy.calls["tax"] += 1
        return self._base.sell_tax(notional)


class SpyCalendar:
    def __init__(self, base: TradingCalendarVN, spy: Spy) -> None:
        self._base = base
        self._spy = spy

    def is_trading_day(self, d) -> bool:
        self._spy.calls["calendar"] += 1
        return self._base.is_trading_day(d)


def test_no_cheating_backtest_and_paper_use_same_dependencies() -> None:
    spy_bt = Spy()
    spy_pp = Spy()

    def build_pipeline(spy: Spy) -> UnifiedTradingPipeline:
        rules = SpyMarketRules(load_market_rules("configs/market_rules_vn.yaml"), spy)
        fees = SpyFees(FeesTaxes(0.001, 0.05, 0.0015, {}), spy)
        cal = SpyCalendar(TradingCalendarVN("configs/trading_calendar_vn.yaml"), spy)

        def cost(px: float) -> float:
            spy.calls["cost"] += 1
            return 0.0005

        def ca(event: dict, positions: dict) -> None:
            del event, positions
            spy.calls["ca"] += 1

        return UnifiedTradingPipeline(
            market_rules=rules,
            fees_taxes=fees,
            cost_model=cost,
            corporate_action_ledger=ca,
            calendar_vn=cal,
        )

    events = [
        {"id": 1, "ts_utc": dt.datetime(2025, 1, 6, 9, 0), "event_type": "signal", "symbol": "AAA", "payload_json": {"symbol": "AAA", "target": 1}},
        {"id": 2, "ts_utc": dt.datetime(2025, 1, 6, 9, 1), "event_type": "bar", "symbol": "AAA", "payload_json": {"symbol": "AAA", "open": 10.0}},
        {"id": 3, "ts_utc": dt.datetime(2025, 1, 7, 9, 0), "event_type": "signal", "symbol": "AAA", "payload_json": {"symbol": "AAA", "target": 0}},
        {"id": 4, "ts_utc": dt.datetime(2025, 1, 7, 9, 1), "event_type": "bar", "symbol": "AAA", "payload_json": {"symbol": "AAA", "open": 11.0}},
    ]

    build_pipeline(spy_bt).run(events, initial_cash=100_000_000.0)

    redis_client = FakeRedisCompat()
    replay_into_redis(
        [EventLog(id=e["id"], ts_utc=e["ts_utc"], source="t", event_type=e["event_type"], symbol=e["symbol"], payload_json=e["payload_json"], payload_hash=f"h{e['id']}") for e in events],
        redis_client,
        speed="max",
    )
    paper_events = []
    for stream in ["ssi:signal", "ssi:bar"]:
        for _, fields in redis_client.xrange(stream):
            paper_events.append(
                {"ts_utc": fields["ts_utc"], "event_type": fields["event_type"], "symbol": fields["symbol"], "payload_json": json.loads(fields["payload"])}
            )

    build_pipeline(spy_pp).run(paper_events, initial_cash=100_000_000.0)

    assert spy_bt.calls == spy_pp.calls
