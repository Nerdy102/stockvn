from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BacktestConfig:
    market: Literal["vn", "crypto"] = "vn"
    trading_type: Literal["spot_paper", "perp_paper"] = "spot_paper"
    position_mode: Literal["long_only", "long_short"] = "long_only"
    execution: Literal["close", "next_bar"] = "close"
    initial_cash: float = 10_000_000.0
    max_position_notional_pct: float = 0.2
    base_bps: float = 5.0
    k_atr: float = 20.0
    k_liq: float = 50.0


@dataclass
class TradeFill:
    entry_time: str
    exit_time: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_px: float
    exit_px: float
    fee: float
    tax: float
    slippage_cost: float
    pnl_gross: float
    pnl_net: float


@dataclass
class EquityPoint:
    time: str
    nav: float
    drawdown: float


@dataclass
class BacktestReportV2:
    metrics: dict[str, float]
    config_hash: str
    dataset_hash: str
    code_hash: str
    report_id: str
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trades: list[TradeFill] = field(default_factory=list)
