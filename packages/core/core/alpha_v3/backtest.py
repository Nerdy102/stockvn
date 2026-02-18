from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from core.cost_model import SlippageConfig
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.portfolio.accounting import PositionAccounting


@dataclass(frozen=True)
class BacktestV3Config:
    initial_cash: float = 1_000_000_000.0
    commission_rate: float = 0.0015
    sell_tax_rate: float = 0.001
    slippage: SlippageConfig = SlippageConfig()
    symbol: str = "TEST"
    random_seed: int = 42


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    return x


def _sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in metrics.items()}


def _align_signal_to_bars(bars: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Support signal indexed by timestamp or by row index."""
    if "timestamp" not in bars.columns:
        raise ValueError("bars must contain timestamp")

    ts = pd.to_datetime(bars["timestamp"])
    sig = pd.Series(signal.copy())

    if isinstance(sig.index, pd.DatetimeIndex):
        out = sig.reindex(ts).fillna(0.0)
        out.index = bars.index
        return out.astype(float)

    # fallback positional alignment
    return sig.reindex(bars.index).fillna(0.0).astype(float)


def _prev_limits(row_prev: pd.Series, market_rules: MarketRules) -> tuple[float, float]:
    ceiling = _safe_float(row_prev.get("ceiling_price", np.nan), np.nan)
    floor = _safe_float(row_prev.get("floor_price", np.nan), np.nan)
    if np.isfinite(ceiling) and np.isfinite(floor):
        return ceiling, floor

    ref = _safe_float(row_prev.get("ref_price", np.nan), np.nan)
    if not np.isfinite(ref) or ref <= 0:
        ref = _safe_float(row_prev.get("close", np.nan), np.nan)
    if not np.isfinite(ref) or ref <= 0:
        return np.nan, np.nan
    low, high = market_rules.calc_price_limits(ref)
    return float(high), float(low)


def _ac_weights(n_days: int, decay: float = 0.35) -> np.ndarray:
    """Monotonic front-loaded trajectory proxy for AC slicing."""
    if n_days <= 0:
        return np.array([], dtype=float)
    idx = np.arange(n_days, dtype=float)
    raw = np.exp(-decay * idx)
    return raw / max(raw.sum(), 1e-12)


def _allocate_board_lot(total_qty: int, weights: np.ndarray, lot_size: int = 100) -> list[int]:
    if total_qty <= 0 or len(weights) == 0:
        return [0] * len(weights)
    lots_total = max(0, total_qty // lot_size)
    if lots_total == 0:
        return [0] * len(weights)
    normalized = weights / max(float(weights.sum()), 1e-12)
    raw_lots = normalized * lots_total
    base_lots = np.floor(raw_lots).astype(int)
    remainder = int(lots_total - int(base_lots.sum()))
    if remainder > 0:
        frac = raw_lots - base_lots
        order = np.argsort(-frac)
        for i in order[:remainder]:
            base_lots[i] += 1
    return [int(v) * lot_size for v in base_lots.tolist()]


def run_backtest_v3(
    bars: pd.DataFrame,
    signal: pd.Series,
    market_rules: MarketRules,
    cfg: BacktestV3Config = BacktestV3Config(),
) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Daily HOSE-aware backtest: signal(t close) -> execute(t+1 open)."""
    d = bars.copy().sort_values("timestamp").reset_index(drop=True)
    d["signal"] = _align_signal_to_bars(d, signal)
    _ = np.random.default_rng(cfg.random_seed)  # fixed seed for deterministic tie-break extensions

    fees = FeesTaxes(cfg.sell_tax_rate, 0.05, cfg.commission_rate, {})
    account = PositionAccounting()
    cash = float(cfg.initial_cash)
    prev_equity = cash
    buy_flows = 0.0
    sell_flows = 0.0
    buy_qty_flow = 0
    sell_qty_flow = 0

    trades: list[dict[str, Any]] = []
    execution_schedules: list[dict[str, Any]] = []
    realized_fills: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    cost_totals = {"commission": 0.0, "sell_tax": 0.0, "slippage": 0.0}

    active_order: dict[str, Any] | None = None
    order_seq = 0

    for i, row in d.iterrows():
        trade_date = pd.to_datetime(row["timestamp"]).date()

        if i > 0 and active_order is None:
            prev = d.iloc[i - 1]
            desired = int(np.sign(_safe_float(prev["signal"], 0.0)))
            side: str | None = None
            order_qty = 0

            if desired > 0 and account.quantity == 0:
                side = "BUY"
                order_qty = int(np.floor(max(cash, 0.0) / max(_safe_float(row["open"], 0.0), 1e-12)))
            elif desired <= 0 and account.quantity > 0:
                side = "SELL"
                order_qty = int(account.quantity)

            if side and order_qty > 0:
                order_qty = (int(order_qty) // 100) * 100
                if order_qty > 0:
                    order_seq += 1
                    plan_qty = _allocate_board_lot(order_qty, _ac_weights(n_days=3), lot_size=100)
                    active_order = {
                        "order_id": f"{cfg.symbol}-{order_seq}",
                        "signal_date": pd.to_datetime(prev["timestamp"]).date(),
                        "side": side,
                        "order_qty": order_qty,
                        "remaining_qty": order_qty,
                        "plan_qty": plan_qty,
                        "slice_idx": 0,
                        "carry_qty": 0,
                    }

        if active_order is not None and active_order["slice_idx"] < 3:
            prev = d.iloc[i - 1] if i > 0 else row
            side = str(active_order["side"])
            slice_idx = int(active_order["slice_idx"])
            open_px = _safe_float(row["open"], 0.0)
            close_prev = _safe_float(prev["close"], 0.0)
            adtv = max(1.0, _safe_float(prev.get("value_vnd", 1e9), 1e9))
            atr14 = _safe_float(prev.get("atr14", abs(prev["high"] - prev["low"])), 0.0)
            spread_proxy = max(0.0, _safe_float(prev.get("spread_proxy", 0.0), 0.0))

            planned_qty = int(active_order["plan_qty"][slice_idx])
            carry_in = int(active_order["carry_qty"])
            intended_qty = min(int(active_order["remaining_qty"]), planned_qty + carry_in)

            cap_notional = 0.05 * adtv
            cap_qty = (int(np.floor(cap_notional / max(open_px, 1e-12))) // 100) * 100
            qty_after_cap = min(intended_qty, max(0, cap_qty))

            part = (float(qty_after_cap) * open_px / adtv) if adtv > 0 else 1.0
            atr_ratio = atr14 / max(close_prev, 1e-12)
            bps_base = float(cfg.slippage.base_bps)
            bps_participation = 50.0 * max(0.0, part)
            bps_atr = 100.0 * max(0.0, atr_ratio)
            bps_spread = spread_proxy * 10000.0
            slp_bps = max(0.0, bps_base + bps_participation + bps_atr + bps_spread)

            slp = slp_bps / 10000.0
            exec_raw = open_px * (1.0 + slp if side == "BUY" else 1.0 - slp)
            exec_px = market_rules.round_price(exec_raw, direction=("up" if side == "BUY" else "down"))

            ceiling_prev, floor_prev = _prev_limits(prev, market_rules)
            limit_hit = (side == "BUY" and np.isfinite(ceiling_prev) and abs(close_prev - ceiling_prev) < 1e-9) or (
                side == "SELL" and np.isfinite(floor_prev) and abs(close_prev - floor_prev) < 1e-9
            )
            fill_ratio = 0.2 if limit_hit else 1.0

            filled_qty = (int(np.floor(qty_after_cap * fill_ratio)) // 100) * 100
            if side == "SELL":
                filled_qty = min(filled_qty, int(account.quantity))

            if side == "BUY" and filled_qty > 0 and exec_px > 0:
                affordable_qty = int(np.floor(cash / max(exec_px * (1.0 + fees.default_commission_rate), 1e-12)))
                filled_qty = min(filled_qty, (affordable_qty // 100) * 100)

            gross_notional = float(filled_qty) * exec_px if filled_qty > 0 else 0.0
            gross_open_notional = float(filled_qty) * open_px if filled_qty > 0 else 0.0
            commission = fees.commission(gross_notional) if filled_qty > 0 else 0.0
            sell_tax = fees.sell_tax(gross_notional) if (filled_qty > 0 and side == "SELL") else 0.0
            slippage_cost = abs(exec_px - open_px) * filled_qty if filled_qty > 0 else 0.0

            if filled_qty > 0 and exec_px > 0:
                if side == "BUY":
                    cash -= gross_notional + commission
                    buy_flows += gross_notional
                    buy_qty_flow += int(filled_qty)
                    account.buy(filled_qty, exec_px, commission)
                else:
                    account.sell(filled_qty, exec_px, commission, sell_tax)
                    cash += gross_notional - commission - sell_tax
                    sell_flows += gross_notional
                    sell_qty_flow += int(filled_qty)

                cost_totals["commission"] += commission
                cost_totals["sell_tax"] += sell_tax
                cost_totals["slippage"] += slippage_cost

                realized_fills.append(
                    {
                        "order_id": active_order["order_id"],
                        "date": trade_date,
                        "symbol": cfg.symbol,
                        "side": side,
                        "slice": slice_idx + 1,
                        "filled_qty": int(filled_qty),
                        "open_price": float(open_px),
                        "exec_price": float(exec_px),
                        "gross_open_notional": float(gross_open_notional),
                        "gross_exec_notional": float(gross_notional),
                        "commission": float(commission),
                        "sell_tax": float(sell_tax),
                        "slippage_cost": float(slippage_cost),
                        "slippage_bps_total": float(slp_bps),
                        "slippage_bps_base": float(bps_base),
                        "slippage_bps_participation": float(bps_participation),
                        "slippage_bps_atr": float(bps_atr),
                        "slippage_bps_spread": float(bps_spread),
                        "fill_ratio": float(fill_ratio),
                    }
                )

            carry_out = max(0, intended_qty - filled_qty)
            active_order["carry_qty"] = int(carry_out)
            active_order["remaining_qty"] = max(0, int(active_order["remaining_qty"]) - int(filled_qty))

            execution_schedules.append(
                {
                    "order_id": active_order["order_id"],
                    "signal_date": active_order["signal_date"],
                    "date": trade_date,
                    "symbol": cfg.symbol,
                    "side": side,
                    "slice": slice_idx + 1,
                    "planned_qty": int(planned_qty),
                    "carry_in_qty": int(carry_in),
                    "intended_qty": int(intended_qty),
                    "cap_qty": int(max(0, cap_qty)),
                    "qty_after_cap": int(max(0, qty_after_cap)),
                    "filled_qty": int(filled_qty),
                    "carry_out_qty": int(carry_out),
                    "unfilled_after_day3": 0,
                }
            )

            trades.append(
                {
                    "date": trade_date,
                    "signal_date": active_order["signal_date"],
                    "symbol": cfg.symbol,
                    "side": side,
                    "order_qty": int(active_order["order_qty"]),
                    "filled_qty": int(filled_qty),
                    "fill_ratio": float(fill_ratio),
                    "open_price": float(open_px),
                    "exec_price": float(exec_px),
                    "slippage_bps": float(slp_bps),
                    "commission": float(commission),
                    "sell_tax": float(sell_tax),
                    "slice": slice_idx + 1,
                    "order_id": active_order["order_id"],
                }
            )

            active_order["slice_idx"] = int(active_order["slice_idx"]) + 1
            if active_order["slice_idx"] >= 3 or active_order["remaining_qty"] <= 0:
                if active_order["remaining_qty"] > 0 and execution_schedules:
                    execution_schedules[-1]["unfilled_after_day3"] = int(active_order["remaining_qty"])
                active_order = None

        close_px = _safe_float(row["close"], 0.0)
        market_value = float(account.quantity) * close_px
        equity = cash + market_value
        daily_return = 0.0 if i == 0 or prev_equity == 0 else equity / prev_equity - 1.0

        if account.quantity < 0:
            raise AssertionError("cash/position reconciliation invariant violated: negative quantity")
        if abs(equity - (cash + market_value)) > 1e-6:
            raise AssertionError("cash/position reconciliation invariant violated")

        expected_cash = cfg.initial_cash - buy_flows + sell_flows - (cost_totals["commission"] + cost_totals["sell_tax"])
        cash_recon_gap = float(cash - expected_cash)
        if abs(cash_recon_gap) > 1.0:
            raise AssertionError("cash flow reconciliation invariant violated (>1 VND)")

        expected_qty = int(buy_qty_flow - sell_qty_flow)
        if int(account.quantity) != expected_qty:
            raise AssertionError("quantity flow reconciliation invariant violated")

        equity_rows.append(
            {
                "date": trade_date,
                "cash": float(cash),
                "position_qty": int(account.quantity),
                "avg_cost": float(account.avg_cost),
                "close": float(close_px),
                "market_value": float(market_value),
                "equity": float(equity),
                "daily_return": float(daily_return),
                "realized_pnl": float(account.realized_pnl),
                "unrealized_pnl": float(account.unrealized_pnl(close_px)),
                "cash_recon_gap": float(cash_recon_gap),
            }
        )
        prev_equity = float(equity)

    if active_order is not None and int(active_order.get("remaining_qty", 0)) > 0 and execution_schedules:
        for idx in range(len(execution_schedules) - 1, -1, -1):
            if execution_schedules[idx].get("order_id") == active_order.get("order_id"):
                execution_schedules[idx]["unfilled_after_day3"] = int(active_order["remaining_qty"])
                break

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    schedule_df = pd.DataFrame(execution_schedules)
    fills_df = pd.DataFrame(realized_fills)

    returns = equity_df["daily_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    downside = returns[returns < 0]
    years = max(len(equity_df) / 252.0, 1e-12)
    cagr = (equity_df["equity"].iloc[-1] / max(cfg.initial_cash, 1e-12)) ** (1.0 / years) - 1.0
    vol = returns.std(ddof=0)
    downside_vol = downside.std(ddof=0)
    sharpe = 0.0 if vol == 0 else (returns.mean() / vol) * sqrt(252.0)
    sortino = 0.0 if downside_vol == 0 else (returns.mean() / downside_vol) * sqrt(252.0)
    running_max = equity_df["equity"].cummax()
    mdd = float((equity_df["equity"] / running_max - 1.0).min()) if len(equity_df) else 0.0

    turnover = 0.0
    if not trades_df.empty:
        traded_notional = float((trades_df["filled_qty"] * trades_df["exec_price"]).sum())
        turnover = traded_notional / max(float(equity_df["equity"].mean()), 1e-12)

    metrics = _sanitize_metrics(
        {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(mdd),
            "turnover": float(turnover),
            "commission_total": float(cost_totals["commission"]),
            "sell_tax_total": float(cost_totals["sell_tax"]),
            "slippage_total": float(cost_totals["slippage"]),
            "realized_pnl": float(account.realized_pnl),
            "final_equity": float(equity_df["equity"].iloc[-1]),
            "cash_recon_gap_abs": float(abs(equity_df["cash_recon_gap"].iloc[-1])) if len(equity_df) else 0.0,
        }
    )

    run_payload = {
        "symbol": cfg.symbol,
        "initial_cash": cfg.initial_cash,
        "commission_rate": cfg.commission_rate,
        "sell_tax_rate": cfg.sell_tax_rate,
        "slippage": {
            "base_bps": cfg.slippage.base_bps,
            "k1": cfg.slippage.k1,
            "k2": cfg.slippage.k2,
        },
        "seed": cfg.random_seed,
        "timestamps": pd.to_datetime(d["timestamp"]).astype(str).tolist(),
        "signal": d["signal"].round(8).tolist(),
        "close": d["close"].round(4).tolist(),
    }
    run_sig = json.dumps(run_payload, sort_keys=True)
    run_hash = sha1(run_sig.encode("utf-8")).hexdigest()[:16]

    backtest_runs = pd.DataFrame(
        [
            {
                "run_hash": run_hash,
                "config_json": {
                    "initial_cash": cfg.initial_cash,
                    "commission_rate": cfg.commission_rate,
                    "sell_tax_rate": cfg.sell_tax_rate,
                    "random_seed": cfg.random_seed,
                },
                "summary_json": metrics,
            }
        ]
    )
    backtest_metrics = pd.DataFrame(
        [{"run_hash": run_hash, "metric_name": k, "metric_value": v} for k, v in metrics.items()]
    )
    backtest_equity_curve = equity_df[["date", "equity"]].copy()
    backtest_equity_curve["run_hash"] = run_hash

    return {
        "trades": trades_df,
        "execution_schedules": schedule_df,
        "realized_fills": fills_df,
        "equity_curve": equity_df,
        "metrics": metrics,
        "backtest_runs": backtest_runs,
        "backtest_metrics": backtest_metrics,
        "backtest_equity_curve": backtest_equity_curve,
    }
