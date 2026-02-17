from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.fees_taxes import FeesTaxes
from core.risk import annualized_volatility, beta, correlation_matrix, historical_var, max_drawdown


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float


def compute_positions_avg_cost(
    trades: pd.DataFrame,
    latest_prices: dict[str, float],
    fees_taxes: FeesTaxes,
    broker_name: str = "demo_broker",
) -> tuple[dict[str, Position], pd.DataFrame]:
    """Average cost basis positions + realized P&L.

    FIXED: SELL clamp quantity *before* notional/commission/tax to avoid fee/tax on wrong qty.
    """
    t = trades.copy()
    t["trade_date"] = pd.to_datetime(t["trade_date"]).dt.date
    t["side"] = t["side"].astype(str).str.upper()
    t["strategy_tag"] = t.get("strategy_tag", "").astype(str)
    t = (
        t.sort_values(["trade_date", "id"], na_position="last")
        if "id" in t.columns
        else t.sort_values(["trade_date"])
    )

    state: dict[str, dict[str, float]] = {}
    realized_rows: list[dict[str, Any]] = []

    for _, row in t.iterrows():
        sym = str(row["symbol"])
        side = str(row["side"])
        qty_req = float(row["quantity"])
        px = float(row["price"])

        state.setdefault(sym, {"qty": 0.0, "avg_cost": 0.0})

        if side == "BUY":
            qty = max(0.0, qty_req)
            notional = qty * px
            comm = fees_taxes.commission(notional, broker_name)

            old_qty = state[sym]["qty"]
            old_cost = state[sym]["avg_cost"]
            new_qty = old_qty + qty
            if new_qty <= 0:
                continue
            new_cost = (old_qty * old_cost + notional + comm) / new_qty
            state[sym]["qty"] = new_qty
            state[sym]["avg_cost"] = new_cost

        elif side == "SELL":
            old_qty = state[sym]["qty"]
            qty = min(max(0.0, qty_req), old_qty)  # clamp first
            if qty <= 0:
                continue

            notional = qty * px
            comm = fees_taxes.commission(notional, broker_name)
            sell_tax = fees_taxes.sell_tax(notional)

            avg_cost = state[sym]["avg_cost"]
            pnl = (px - avg_cost) * qty - comm - sell_tax

            state[sym]["qty"] = max(0.0, old_qty - qty)

            realized_rows.append(
                {
                    "trade_date": row["trade_date"],
                    "symbol": sym,
                    "strategy_tag": str(row.get("strategy_tag", "")),
                    "quantity": qty,
                    "sell_price": px,
                    "avg_cost": avg_cost,
                    "commission": comm,
                    "sell_tax": sell_tax,
                    "realized_pnl": pnl,
                }
            )
        else:
            raise ValueError(f"Unknown side: {side}")

    positions: dict[str, Position] = {}
    for sym, st in state.items():
        qty = float(st["qty"])
        if qty <= 0:
            continue
        mp = float(latest_prices.get(sym, np.nan))
        mv = qty * mp if not np.isnan(mp) else 0.0
        upnl = (mp - float(st["avg_cost"])) * qty if not np.isnan(mp) else 0.0
        positions[sym] = Position(
            symbol=sym,
            quantity=qty,
            avg_cost=float(st["avg_cost"]),
            market_price=mp,
            market_value=mv,
            unrealized_pnl=upnl,
        )

    realized = pd.DataFrame(realized_rows)
    return positions, realized


def _simulate_cash_holdings(
    trades: pd.DataFrame,
    price_panel: pd.DataFrame,
    cash_start: float,
    fees_taxes: FeesTaxes,
    broker_name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate daily cash & holdings (units) across price_panel.index.

    FIXED: SELL clamp before notional/fees/taxes.
    """
    t = trades.copy()
    t["trade_date"] = pd.to_datetime(t["trade_date"]).dt.date
    t["side"] = t["side"].astype(str).str.upper()
    t = (
        t.sort_values(["trade_date", "id"], na_position="last")
        if "id" in t.columns
        else t.sort_values(["trade_date"])
    )

    dates = list(price_panel.index)
    holdings: dict[str, float] = {c: 0.0 for c in price_panel.columns}
    cash = float(cash_start)
    cash_series: list[float] = []
    hold_rows: list[dict[str, float]] = []

    by_date: dict[Any, pd.DataFrame] = {d: df for d, df in t.groupby("trade_date")}

    for d in dates:
        if d in by_date:
            for _, r in by_date[d].iterrows():
                sym = str(r["symbol"])
                side = str(r["side"])
                qty_req = float(r["quantity"])
                px = float(r["price"])
                if sym not in holdings:
                    holdings[sym] = 0.0

                if side == "BUY":
                    qty = max(0.0, qty_req)
                    notional = qty * px
                    comm = fees_taxes.commission(notional, broker_name)
                    holdings[sym] += qty
                    cash -= notional + comm

                elif side == "SELL":
                    old_qty = holdings.get(sym, 0.0)
                    qty = min(max(0.0, qty_req), old_qty)  # clamp first
                    if qty <= 0:
                        continue
                    notional = qty * px
                    comm = fees_taxes.commission(notional, broker_name)
                    tax = fees_taxes.sell_tax(notional)
                    holdings[sym] = max(0.0, old_qty - qty)
                    cash += notional - comm - tax

        cash_series.append(cash)
        hold_rows.append(dict(holdings))

    cash_s = pd.Series(cash_series, index=price_panel.index, name="cash")
    hold_df = pd.DataFrame(hold_rows, index=price_panel.index).fillna(0.0)
    return cash_s, hold_df


def infer_start_cash(
    trades: pd.DataFrame, fees_taxes: FeesTaxes, broker_name: str = "demo_broker"
) -> float:
    """Infer a non-negative starting cash so that cash does not go negative (MVP helper)."""
    t = trades.copy()
    if t.empty:
        return 0.0
    t["trade_date"] = pd.to_datetime(t["trade_date"]).dt.date
    t["side"] = t["side"].astype(str).str.upper()
    t = (
        t.sort_values(["trade_date", "id"], na_position="last")
        if "id" in t.columns
        else t.sort_values(["trade_date"])
    )
    cash = 0.0
    min_cash = 0.0
    positions: dict[str, float] = {}
    for _, r in t.iterrows():
        sym = str(r["symbol"])
        side = str(r["side"])
        qty_req = float(r["quantity"])
        px = float(r["price"])
        positions.setdefault(sym, 0.0)

        if side == "BUY":
            qty = max(0.0, qty_req)
            notional = qty * px
            comm = fees_taxes.commission(notional, broker_name)
            positions[sym] += qty
            cash -= notional + comm
        elif side == "SELL":
            old_qty = positions.get(sym, 0.0)
            qty = min(max(0.0, qty_req), old_qty)
            if qty <= 0:
                continue
            notional = qty * px
            comm = fees_taxes.commission(notional, broker_name)
            tax = fees_taxes.sell_tax(notional)
            positions[sym] = max(0.0, old_qty - qty)
            cash += notional - comm - tax
        min_cash = min(min_cash, cash)
    # Add small buffer
    return float(max(0.0, -min_cash) * 1.02)


def compute_portfolio_value_series(
    trades: pd.DataFrame,
    price_panel: pd.DataFrame,
    cash_start: float,
    fees_taxes: FeesTaxes,
    broker_name: str = "demo_broker",
) -> tuple[pd.Series, pd.Series]:
    """Daily portfolio value series (MVP). Returns (value_series, cash_series)."""
    cash_s, hold_df = _simulate_cash_holdings(
        trades, price_panel, cash_start, fees_taxes, broker_name
    )
    mv = (hold_df * price_panel).sum(axis=1)
    val = (cash_s + mv).rename("portfolio_value")
    return val, cash_s


def time_weighted_return(value_series: pd.Series) -> float:
    v = value_series.dropna()
    if len(v) < 2:
        return 0.0
    return float(v.iloc[-1] / v.iloc[0] - 1.0)


def portfolio_risk_summary(
    portfolio_values: pd.Series, benchmark_close: pd.Series
) -> dict[str, float]:
    pv = portfolio_values.dropna()
    if len(pv) < 10:
        return {"volatility": 0.0, "max_drawdown": 0.0, "beta": 0.0, "var_95": 0.0}
    pret = pv.pct_change().dropna()
    bret = benchmark_close.reindex(pv.index).pct_change().dropna()
    return {
        "volatility": annualized_volatility(pret),
        "max_drawdown": max_drawdown(pv),
        "beta": beta(pret, bret),
        "var_95": historical_var(pret, alpha=0.05),
    }


def exposure_by_sector(positions: dict[str, Position], tickers: pd.DataFrame) -> dict[str, float]:
    """Exposure by sector based on current market value."""
    if not positions:
        return {}
    total = sum(p.market_value for p in positions.values())
    if total <= 0:
        return {}
    sec_map = tickers.set_index("symbol")["sector"].to_dict()
    out: dict[str, float] = {}
    for sym, pos in positions.items():
        sec = str(sec_map.get(sym, "Unknown"))
        out[sec] = out.get(sec, 0.0) + (pos.market_value / total)
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def concentration_metrics(positions: dict[str, Position]) -> dict[str, Any]:
    if not positions:
        return {"top_weights": [], "hhi": 0.0}
    mv = pd.Series({k: v.market_value for k, v in positions.items()})
    total = float(mv.sum())
    if total <= 0:
        return {"top_weights": [], "hhi": 0.0}
    w = mv / total
    top = w.sort_values(ascending=False).head(5)
    hhi = float((w**2).sum())
    return {
        "top_weights": [{"symbol": s, "weight": float(wt)} for s, wt in top.items()],
        "hhi": hhi,
    }


def realized_pnl_breakdown(realized: pd.DataFrame) -> dict[str, Any]:
    if realized.empty:
        return {"by_day": [], "by_strategy": []}
    r = realized.copy()
    r["trade_date"] = pd.to_datetime(r["trade_date"]).dt.date
    by_day = r.groupby("trade_date")["realized_pnl"].sum().sort_index()
    by_strat = r.groupby("strategy_tag")["realized_pnl"].sum().sort_values(ascending=False)
    return {
        "by_day": [{"trade_date": str(k), "realized_pnl": float(v)} for k, v in by_day.items()],
        "by_strategy": [
            {"strategy_tag": str(k), "realized_pnl": float(v)} for k, v in by_strat.items()
        ],
    }


def correlation_of_holdings(price_panel: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    p = price_panel.dropna(axis=1, how="all").tail(lookback)
    if p.shape[0] < 3 or p.shape[1] < 2:
        return pd.DataFrame()
    rets = p.pct_change().dropna(how="all")
    return correlation_matrix(rets)


def brinson_attribution_mvp(
    w_port: pd.Series,
    r_port: pd.Series,
    w_bench: pd.Series,
    r_bench: pd.Series,
) -> dict[str, float]:
    """Brinson-style attribution (MVP).

    - Allocation effect: Σ (Wp - Wb) * (Rb - Rb_total)
    - Selection effect:  Σ Wb * (Rp - Rb)
    - Interaction:       Σ (Wp - Wb) * (Rp - Rb)

    Mapping:
    - market ~ benchmark total return
    - selection ~ selection effect
    - timing_proxy ~ allocation + interaction (MVP proxy)
    """
    sectors = sorted(
        set(w_port.index) | set(w_bench.index) | set(r_port.index) | set(r_bench.index)
    )
    w_p = w_port.reindex(sectors).fillna(0.0)
    w_b = w_bench.reindex(sectors).fillna(0.0)
    r_p = r_port.reindex(sectors).fillna(0.0)
    r_b = r_bench.reindex(sectors).fillna(0.0)

    r_b_total = float((w_b * r_b).sum())
    allocation = float(((w_p - w_b) * (r_b - r_b_total)).sum())
    selection = float((w_b * (r_p - r_b)).sum())
    interaction = float(((w_p - w_b) * (r_p - r_b)).sum())

    return {
        "benchmark_return": r_b_total,
        "allocation_effect": allocation,
        "selection_effect": selection,
        "interaction_effect": interaction,
        "timing_proxy": allocation + interaction,
        "active_return": allocation + selection + interaction,
    }


def suggest_rebalance(
    positions: dict[str, Position],
    tickers: pd.DataFrame,
    cash: float,
    rules: dict[str, float] | None = None,
    regime_state: str = "sideway",
) -> dict[str, Any]:
    """Risk-aware rebalance suggestion (MVP).

    rules:
      - target_cash_weight (default 0.10)
      - max_single_name_weight (default 0.10)
      - max_sector_weight (default 0.25)
      - board_lot (default 100)
      - beta_cap (default 1.1)
      - participation_limit (default 0.05)
      - days_to_exit (default 3)
      - min_signal_strength (default 0.0)
    """
    rules = rules or {}
    target_cash = float(rules.get("target_cash_weight", 0.10))
    max_name = float(rules.get("max_single_name_weight", 0.10))
    max_sector = float(rules.get("max_sector_weight", 0.25))
    board_lot = int(rules.get("board_lot", 100))
    beta_cap = float(rules.get("beta_cap", 1.1))
    participation_limit = float(rules.get("participation_limit", 0.05))
    days_to_exit = float(rules.get("days_to_exit", 3.0))

    if not positions:
        return {"rules": rules, "suggestions": [], "note": "No positions."}

    sec_map = tickers.set_index("symbol")["sector"].to_dict()
    beta_map = tickers.set_index("symbol").get("beta", pd.Series(dtype=float)).to_dict()
    adtv_map = tickers.set_index("symbol").get("adtv_20d", pd.Series(dtype=float)).to_dict()

    mv = pd.Series({s: p.market_value for s, p in positions.items()})
    total = float(mv.sum() + cash)
    if total <= 0:
        total = float(mv.sum())
    if total <= 0:
        return {"rules": rules, "suggestions": [], "note": "No market value."}

    # Regime overlay: risk-off => higher cash buffer
    if regime_state == "risk_off":
        target_cash = max(target_cash, 0.2)

    w_name = mv / total
    w_sector = w_name.groupby(w_name.index.map(lambda s: sec_map.get(s, "Unknown"))).sum()

    suggestions: list[dict[str, Any]] = []
    cash_w = float(cash / total)
    need_cash = max(0.0, target_cash - cash_w) * total

    def _notional_to_qty(sym: str, notional: float) -> float:
        px = float(positions[sym].market_price or 0.0)
        if px <= 0:
            return 0.0
        q = int(notional / px)
        return float((q // board_lot) * board_lot)

    # Step 1: Reduce overweight names (survival-first)
    for sym, wt in w_name.sort_values(ascending=False).items():
        if wt > max_name:
            excess = (wt - max_name) * total
            qty = _notional_to_qty(sym, excess)
            if qty <= 0:
                continue
            notional = qty * float(positions[sym].market_price)
            adtv = float(adtv_map.get(sym, 0.0) or 0.0)
            liq_cap = adtv * participation_limit * days_to_exit if adtv > 0 else notional
            notional = min(notional, liq_cap)
            suggestions.append(
                {
                    "action": "SELL",
                    "symbol": sym,
                    "quantity": qty,
                    "notional_vnd": float(notional),
                    "reason": "max_single_name_weight",
                    "odd_lot_flag": bool(qty % board_lot != 0),
                }
            )

    # Step 2: Reduce overweight sectors
    for sec, wt in w_sector.sort_values(ascending=False).items():
        if wt > max_sector:
            excess = (wt - max_sector) * total
            syms = [s for s in w_name.index if sec_map.get(s, "Unknown") == sec]
            for s_ in syms:
                part = float(w_name[s_] / wt) if wt > 0 else 0.0
                target_notional = excess * part
                qty = _notional_to_qty(s_, target_notional)
                if qty <= 0:
                    continue
                suggestions.append(
                    {
                        "action": "SELL",
                        "symbol": s_,
                        "quantity": qty,
                        "notional_vnd": float(qty * positions[s_].market_price),
                        "reason": "max_sector_weight",
                        "odd_lot_flag": bool(qty % board_lot != 0),
                    }
                )

    # Step 3: Raise cash if needed
    if need_cash > 0:
        for sym, wt in w_name.sort_values(ascending=False).items():
            if need_cash <= 0:
                break
            take = min(need_cash, float(wt * total * 0.25))
            qty = _notional_to_qty(sym, take)
            if qty <= 0:
                continue
            notional = float(qty * positions[sym].market_price)
            suggestions.append(
                {
                    "action": "SELL",
                    "symbol": sym,
                    "quantity": qty,
                    "notional_vnd": notional,
                    "reason": "target_cash_weight",
                    "odd_lot_flag": bool(qty % board_lot != 0),
                }
            )
            need_cash -= notional

    # Beta cap diagnostic (portfolio-level)
    w = (mv / mv.sum()) if mv.sum() > 0 else pd.Series(dtype=float)
    port_beta = (
        float(sum(float(w.get(sym, 0.0)) * float(beta_map.get(sym, 1.0) or 1.0) for sym in w.index))
        if not w.empty
        else 0.0
    )

    note = "Heuristic suggestions include board-lot rounding, liquidity caps, cash buffer and risk overlay."
    return {
        "rules": {
            "target_cash_weight": target_cash,
            "max_single_name_weight": max_name,
            "max_sector_weight": max_sector,
            "board_lot": board_lot,
            "beta_cap": beta_cap,
            "participation_limit": participation_limit,
            "days_to_exit": days_to_exit,
        },
        "diagnostics": {
            "regime_state": regime_state,
            "portfolio_beta_est": port_beta,
            "beta_cap_breached": bool(port_beta > beta_cap),
        },
        "suggestions": suggestions,
        "note": note,
    }
