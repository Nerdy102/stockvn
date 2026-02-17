from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.cost_model import SlippageConfig, apply_execution_slippage, calc_slippage_bps
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000_000.0
    commission_rate: float = 0.0015
    sell_tax_rate: float = 0.001
    slippage: SlippageConfig = SlippageConfig()


def run_next_bar_backtest(
    bars: pd.DataFrame,
    signal: pd.Series,
    market_rules: MarketRules,
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Long-only MVP backtest: signal(t) executes at next bar open(t+1)."""
    d = bars.copy().sort_values("timestamp")
    d["signal"] = signal.reindex(d.index).fillna(False).astype(bool)
    d["next_open"] = d["open"].shift(-1)
    d["position"] = d["signal"].shift(1).fillna(False).astype(int)

    fees = FeesTaxes(cfg.sell_tax_rate, 0.05, cfg.commission_rate, {})
    rets = []
    for _, r in d.iloc[:-1].iterrows():
        if int(r["position"]) == 0:
            rets.append(0.0)
            continue
        px = float(r["next_open"])
        slip = calc_slippage_bps(
            order_notional=1e8,
            adtv=max(1.0, float(r.get("value_vnd", 1e9))),
            atr14=abs(float(r["high"] - r["low"])),
            close=max(float(r["close"]), 1.0),
            cfg=cfg.slippage,
        )
        buy_px = apply_execution_slippage(px, "BUY", slip)
        sell_px = apply_execution_slippage(float(r["close"]), "SELL", slip)
        buy_px = market_rules.round_price(buy_px, direction="up")
        sell_px = market_rules.round_price(sell_px, direction="down")
        gross = (sell_px - buy_px) / max(buy_px, 1e-9)
        comm = fees.commission(1.0, None) * 2
        tax = fees.sell_tax(max(0.0, sell_px))
        rets.append(gross - comm - tax)

    out = d.iloc[:-1][["timestamp", "close"]].copy()
    out["strategy_ret"] = rets
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod() * cfg.initial_cash
    return out
