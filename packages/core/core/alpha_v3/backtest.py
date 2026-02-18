from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from core.cost_model import SlippageConfig, calc_slippage_bps
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

    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    cost_totals = {"commission": 0.0, "sell_tax": 0.0, "slippage": 0.0}

    for i, row in d.iterrows():
        trade_date = pd.to_datetime(row["timestamp"]).date()

        if i > 0:
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
                open_px = _safe_float(row["open"], 0.0)
                adtv = max(1.0, _safe_float(prev.get("value_vnd", 1e9), 1e9))
                atr14 = _safe_float(prev.get("atr14", abs(prev["high"] - prev["low"])), 0.0)
                close_prev = _safe_float(prev["close"], 0.0)

                slp_bps = calc_slippage_bps(
                    order_notional=float(order_qty) * max(open_px, 1e-12),
                    adtv=adtv,
                    atr14=atr14,
                    close=max(close_prev, 1e-12),
                    cfg=cfg.slippage,
                )
                slp = slp_bps / 10000.0
                exec_raw = open_px * (1.0 + slp if side == "BUY" else 1.0 - slp)
                exec_px = market_rules.round_price(exec_raw, direction=("up" if side == "BUY" else "down"))

                ceiling_prev, floor_prev = _prev_limits(prev, market_rules)
                fill_ratio = 1.0
                if side == "BUY" and np.isfinite(ceiling_prev) and abs(close_prev - ceiling_prev) < 1e-9:
                    fill_ratio *= 0.2
                if side == "SELL" and np.isfinite(floor_prev) and abs(close_prev - floor_prev) < 1e-9:
                    fill_ratio *= 0.2

                filled_qty = int(np.floor(order_qty * fill_ratio))
                filled_qty = (filled_qty // 100) * 100
                if side == "SELL":
                    # clamp before fees/taxes
                    filled_qty = min(filled_qty, int(account.quantity))

                if side == "BUY" and filled_qty > 0 and exec_px > 0:
                    affordable_qty = int(
                        np.floor(cash / max(exec_px * (1.0 + fees.default_commission_rate), 1e-12))
                    )
                    filled_qty = min(filled_qty, (affordable_qty // 100) * 100)

                if filled_qty > 0 and exec_px > 0:
                    gross_notional = float(filled_qty) * exec_px
                    commission = fees.commission(gross_notional)
                    sell_tax = fees.sell_tax(gross_notional) if side == "SELL" else 0.0

                    if side == "BUY":
                        cash -= gross_notional + commission
                        account.buy(filled_qty, exec_px, commission)
                    else:
                        account.sell(filled_qty, exec_px, commission, sell_tax)
                        cash += gross_notional - commission - sell_tax

                    cost_totals["commission"] += commission
                    cost_totals["sell_tax"] += sell_tax
                    cost_totals["slippage"] += abs(exec_px - open_px) * filled_qty

                    trades.append(
                        {
                            "date": trade_date,
                            "signal_date": pd.to_datetime(prev["timestamp"]).date(),
                            "symbol": cfg.symbol,
                            "side": side,
                            "order_qty": int(order_qty),
                            "filled_qty": int(filled_qty),
                            "fill_ratio": float(fill_ratio),
                            "open_price": float(open_px),
                            "exec_price": float(exec_px),
                            "slippage_bps": float(slp_bps),
                            "commission": float(commission),
                            "sell_tax": float(sell_tax),
                        }
                    )

        close_px = _safe_float(row["close"], 0.0)
        market_value = float(account.quantity) * close_px
        equity = cash + market_value
        daily_return = 0.0 if i == 0 or prev_equity == 0 else equity / prev_equity - 1.0

        if account.quantity < 0:
            raise AssertionError("cash/position reconciliation invariant violated: negative quantity")
        if abs(equity - (cash + market_value)) > 1e-6:
            raise AssertionError("cash/position reconciliation invariant violated")

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
            }
        )
        prev_equity = float(equity)

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

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
        "equity_curve": equity_df,
        "metrics": metrics,
        "backtest_runs": backtest_runs,
        "backtest_metrics": backtest_metrics,
        "backtest_equity_curve": backtest_equity_curve,
    }
