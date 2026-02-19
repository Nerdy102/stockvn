from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Callable

import pandas as pd

from core.backtest_v2.costs import apply_price_with_slippage, slippage_bps
from core.backtest_v2.metrics import compute_metrics
from core.backtest_v2.schemas import BacktestConfig, BacktestReportV2, EquityPoint, TradeFill


def _hash_obj(v: object) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode("utf-8")).hexdigest()[
        :16
    ]


def _code_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:16]
    except Exception:
        return "unknown_code_hash"


def run_backtest_v2(
    df: pd.DataFrame,
    config: BacktestConfig,
    signal_fn: Callable[[pd.DataFrame], str],
    *,
    fee_rate: float,
    sell_tax_rate: float,
    include_equity_curve: bool = False,
    include_trades: bool = False,
) -> BacktestReportV2:
    w = df.copy().reset_index(drop=True)
    if "timestamp" not in w.columns and "date" in w.columns:
        w["timestamp"] = w["date"]
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")

    cash = config.initial_cash
    nav = [cash]
    eq: list[EquityPoint] = []
    trades: list[TradeFill] = []
    pos_side = 0
    entry_px = 0.0
    qty = 0.0
    entry_ts = ""

    for i in range(len(w) - 1):
        hist = w.iloc[: i + 1].copy()
        row = w.iloc[i]
        ts = str(row["timestamp"])
        signal = signal_fn(hist)
        px = float(row["close"])
        atr_pct = 0.01
        dv = float(row.get("close", 0.0) * row.get("volume", 0.0))
        sbps = slippage_bps(config.base_bps, config.k_atr, config.k_liq, atr_pct, dv)

        if config.execution == "next_bar":
            fill_px_raw = float(w.iloc[i + 1]["open"])
            fill_ts = str(w.iloc[i + 1]["timestamp"])
        else:
            fill_px_raw = px
            fill_ts = ts

        if pos_side == 0:
            if signal == "BUY":
                fill_px = apply_price_with_slippage(fill_px_raw, "BUY", sbps)
                notional = cash * config.max_position_notional_pct
                qty = max(notional / max(fill_px, 1e-9), 0.0)
                entry_px = fill_px
                entry_ts = fill_ts
                pos_side = 1
                cash -= qty * fill_px
            elif signal == "SELL" and config.position_mode == "long_short":
                fill_px = apply_price_with_slippage(fill_px_raw, "SELL", sbps)
                notional = cash * config.max_position_notional_pct
                qty = max(notional / max(fill_px, 1e-9), 0.0)
                entry_px = fill_px
                entry_ts = fill_ts
                pos_side = -1
        else:
            should_exit = (pos_side == 1 and signal != "BUY") or (
                pos_side == -1 and signal != "SELL"
            )
            if should_exit:
                exit_side = "SELL" if pos_side == 1 else "BUY"
                exit_px = apply_price_with_slippage(fill_px_raw, exit_side, sbps)
                pnl_gross = (
                    (exit_px - entry_px) * qty if pos_side == 1 else (entry_px - exit_px) * qty
                )
                notional = qty * (entry_px + exit_px)
                fee = notional * fee_rate
                tax = (
                    qty * exit_px * sell_tax_rate
                    if (config.market == "vn" and pos_side == 1)
                    else 0.0
                )
                slip_cost = qty * abs(exit_px - fill_px_raw)
                pnl_net = pnl_gross - fee - tax - slip_cost
                cash += pnl_net + (qty * entry_px if pos_side == 1 else 0.0)
                trades.append(
                    TradeFill(
                        entry_time=entry_ts,
                        exit_time=fill_ts,
                        side="LONG" if pos_side == 1 else "SHORT",
                        qty=qty,
                        entry_px=entry_px,
                        exit_px=exit_px,
                        fee=fee,
                        tax=tax,
                        slippage_cost=slip_cost,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                    )
                )
                pos_side = 0
                qty = 0.0

        mark = qty * px
        nav_val = cash + (mark if pos_side == 1 else 0.0)
        nav.append(nav_val)
        peak = max(nav)
        dd = nav_val / max(peak, 1e-9) - 1.0
        eq.append(EquityPoint(time=ts, nav=nav_val, drawdown=dd))

    config_hash = _hash_obj(config.__dict__)
    dataset_hash = _hash_obj(
        w[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records")
    )
    code_hash = _code_hash()
    report_id = _hash_obj(
        {"config_hash": config_hash, "dataset_hash": dataset_hash, "code_hash": code_hash}
    )
    out = BacktestReportV2(
        metrics=compute_metrics(nav),
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        code_hash=code_hash,
        report_id=report_id,
        equity_curve=eq if include_equity_curve else [],
        trades=trades if include_trades else [],
    )
    return out
