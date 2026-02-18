from __future__ import annotations

import pandas as pd

from core.portfolio import dashboard as dmod


class _DummyRes:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _DummyDB:
    def exec(self, _):
        return _DummyRes([])


def test_trade_explain_reason_keys_fixed(monkeypatch) -> None:
    monkeypatch.setattr(
        dmod,
        "build_portfolio_dashboard",
        lambda db, pid: {
            "holdings": [{"symbol": "AAA", "price": 10.0, "value": 1_000_000.0, "weight": 0.2, "quantity": 100000}],
            "cash": 100_000.0,
        },
    )
    monkeypatch.setattr(
        dmod,
        "suggest_rebalance",
        lambda *args, **kwargs: {
            "suggestions": [
                {"symbol": "AAA", "action": "SELL", "quantity": 1000.0, "reason": "max_single_name_weight"}
            ]
        },
    )
    monkeypatch.setattr(dmod, "load_execution_assumptions", lambda *_: type("A", (), {"base_slippage_bps": 10, "k1_participation": 5, "k2_volatility": 10, "limit_up_buy_fill_ratio": 0.5, "limit_down_sell_fill_ratio": 0.5})())
    monkeypatch.setattr(dmod, "slippage_bps", lambda *_, **__: 10.0)
    monkeypatch.setattr(dmod, "get_settings", lambda: type("S", (), {"FEES_TAXES_PATH": "configs/fees_taxes.yaml", "EXECUTION_MODEL_PATH": "configs/execution_model.yaml", "BROKER_NAME": "demo_broker"})())

    out = dmod.build_rebalance_preview(_DummyDB(), 1)
    assert out["trades"][0]["reason_key"] in dmod.FIXED_REASON_KEYS
    assert set(out["explain_reason_keys"]) == dmod.FIXED_REASON_KEYS
