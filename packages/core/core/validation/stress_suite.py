from __future__ import annotations

import random
from typing import Any, Callable

import pandas as pd

from core.backtest_v3.engine import BacktestV3Config, FundingRatePoint, run_backtest_v3


SCENARIOS = {
    "COST_MULTIPLIER": [(f, s) for f in (1.0, 2.0, 3.0) for s in (1.0, 2.0, 3.0)],
    "EXECUTION_DELAY": [0, 1, 2],
    "SPREAD_STRESS": [1.0, 1.5, 2.0],
    "DATA_GAP": [0.0, 0.01],
    "CRYPTO_FUNDING_STRESS": [-0.0001, 0.0, 0.0001],
}


def _initial_cash(config: BacktestV3Config) -> float:
    return config.initial_cash_vn if config.market == "vn" else config.initial_cash_crypto


def _shift_bars(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if bars <= 0:
        return df
    out = df.copy().reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        out[c] = out[c].shift(bars).bfill()
    return out


def _drop_bars_deterministic(df: pd.DataFrame, drop_rate: float) -> pd.DataFrame:
    if drop_rate <= 0:
        return df
    rng = random.Random(42)
    keep = [rng.random() >= drop_rate for _ in range(len(df))]
    return df.loc[keep].reset_index(drop=True)


def _as_row(scenario: str, params: dict[str, Any], rep: Any, *, costs_total: float | None = None, net_return: float | None = None, mdd: float | None = None, sharpe: float | None = None) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "params": params,
        "net_return": rep.net_return if net_return is None else net_return,
        "mdd": rep.mdd if mdd is None else mdd,
        "sharpe": rep.sharpe if sharpe is None else sharpe,
        "costs_total": sum(rep.costs_breakdown.values()) if costs_total is None else costs_total,
    }


def run_stress_suite(
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: BacktestV3Config,
    signal_fn: Callable[[pd.DataFrame], str],
    fees_taxes_path: str,
    fees_crypto_path: str,
    execution_model_path: str,
) -> dict[str, Any]:
    base = run_backtest_v3(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        config=config,
        signal_fn=signal_fn,
        fees_taxes_path=fees_taxes_path,
        fees_crypto_path=fees_crypto_path,
        execution_model_path=execution_model_path,
    )
    initial_cash = _initial_cash(config)
    base_costs = sum(base.costs_breakdown.values())

    rows: list[dict[str, Any]] = []

    for fee_mul, slip_mul in SCENARIOS["COST_MULTIPLIER"]:
        rep = run_backtest_v3(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            signal_fn=signal_fn,
            fees_taxes_path=fees_taxes_path,
            fees_crypto_path=fees_crypto_path,
            execution_model_path=execution_model_path,
        )
        stressed_costs = (
            rep.costs_breakdown.get("fee_total", 0.0) * fee_mul
            + rep.costs_breakdown.get("tax_total", 0.0)
            + rep.costs_breakdown.get("slippage_total", 0.0) * slip_mul
            + rep.costs_breakdown.get("funding_total", 0.0)
        )
        delta_cost = stressed_costs - sum(rep.costs_breakdown.values())
        stressed_net = rep.net_return - (delta_cost / max(initial_cash, 1e-9))
        rows.append(
            _as_row(
                "COST_MULTIPLIER",
                {"fee_mul": fee_mul, "slippage_mul": slip_mul},
                rep,
                costs_total=stressed_costs,
                net_return=stressed_net,
            )
        )

    for d in SCENARIOS["EXECUTION_DELAY"]:
        rep = run_backtest_v3(
            df=_shift_bars(df, d),
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            signal_fn=signal_fn,
            fees_taxes_path=fees_taxes_path,
            fees_crypto_path=fees_crypto_path,
            execution_model_path=execution_model_path,
        )
        rows.append(_as_row("EXECUTION_DELAY", {"delay_bars": d}, rep))

    for sp in SCENARIOS["SPREAD_STRESS"]:
        rep = run_backtest_v3(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            signal_fn=signal_fn,
            fees_taxes_path=fees_taxes_path,
            fees_crypto_path=fees_crypto_path,
            execution_model_path=execution_model_path,
        )
        rows.append(
            _as_row(
                "SPREAD_STRESS",
                {"spread_multiplier": sp},
                rep,
                costs_total=sum(rep.costs_breakdown.values()) * sp,
                net_return=rep.net_return / max(sp, 1e-9),
                mdd=rep.mdd * sp,
                sharpe=rep.sharpe / max(sp, 1e-9),
            )
        )

    for dr in SCENARIOS["DATA_GAP"]:
        rep = run_backtest_v3(
            df=_drop_bars_deterministic(df, dr),
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            signal_fn=signal_fn,
            fees_taxes_path=fees_taxes_path,
            fees_crypto_path=fees_crypto_path,
            execution_model_path=execution_model_path,
        )
        rows.append(_as_row("DATA_GAP", {"drop_rate": dr}, rep))

    if config.market == "crypto" and config.trading_type == "perp_paper":
        ts_col = "date" if "date" in df.columns else "timestamp"
        for shift in SCENARIOS["CRYPTO_FUNDING_STRESS"]:
            rates = [
                FundingRatePoint(ts=str(ts), symbol=symbol, funding_rate=shift)
                for ts in pd.to_datetime(df[ts_col], errors="coerce").astype(str).tolist()
            ]
            rep = run_backtest_v3(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                config=config,
                signal_fn=signal_fn,
                fees_taxes_path=fees_taxes_path,
                fees_crypto_path=fees_crypto_path,
                execution_model_path=execution_model_path,
                funding_rates=rates,
            )
            rows.append(_as_row("CRYPTO_FUNDING_STRESS", {"funding_rate_shift": shift}, rep))

    worst = (
        min(rows, key=lambda r: r["net_return"])
        if rows
        else {
            "net_return": base.net_return,
            "mdd": base.mdd,
            "sharpe": base.sharpe,
            "costs_total": base_costs,
        }
    )
    sensitivity = (base.net_return - float(worst["net_return"])) / abs(base.net_return + 1e-9)
    return {
        "base": {
            "net_return": base.net_return,
            "mdd": base.mdd,
            "sharpe": base.sharpe,
            "costs_total": base_costs,
        },
        "stress_table": rows,
        "worst_case": worst,
        "sensitivity_index": float(sensitivity),
    }
