from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from core.db.models import (
    Portfolio,
    PriceOHLCV,
    RebalanceConstraintReport,
    Ticker,
    Trade,
)
from core.execution_model import load_execution_assumptions, slippage_bps
from core.fees_taxes import FeesTaxes
from core.portfolio.analytics import (
    compute_portfolio_value_series,
    compute_positions_avg_cost,
    exposure_by_sector,
    infer_start_cash,
    portfolio_risk_summary,
    suggest_rebalance,
)
from core.settings import get_settings

FIXED_REASON_KEYS = {
    "MAX_SINGLE",
    "MAX_SECTOR",
    "MIN_CASH",
    "LIQUIDITY_CAP",
    "CLUSTER_CAP",
    "TURNOVER_CAP",
    "REBALANCE_TO_TARGET",
}


def _load_portfolio_state(db: Session, portfolio_id: int) -> dict[str, Any]:
    settings = get_settings()
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    broker = settings.BROKER_NAME

    trades = db.exec(select(Trade).where(Trade.portfolio_id == portfolio_id)).all()
    tdf = pd.DataFrame([t.model_dump() for t in trades]) if trades else pd.DataFrame()

    prices = db.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    pdf = pd.DataFrame([p.model_dump() for p in prices]) if prices else pd.DataFrame()
    if pdf.empty:
        return {"positions": {}, "panel": pd.DataFrame(), "cash": 0.0, "nav": 0.0, "risk": {}}

    pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date
    pdf = (
        pdf.sort_values(["symbol", "date", "timestamp"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .sort_values(["symbol", "date"])
    )
    latest_prices = pdf.groupby("symbol")["close"].last().to_dict()

    if tdf.empty:
        positions = {}
    else:
        positions, _ = compute_positions_avg_cost(tdf, latest_prices, fees, broker_name=broker)

    panel = pdf.pivot(index="date", columns="symbol", values="close").sort_index()
    port_panel = (
        panel[[c for c in panel.columns if c in positions]].copy()
        if positions
        else pd.DataFrame(index=panel.index)
    )

    if tdf.empty or port_panel.empty:
        cash_now = 0.0
        port_val = pd.Series(dtype=float)
    else:
        cash_start = infer_start_cash(tdf, fees, broker_name=broker)
        port_val, cash_series = compute_portfolio_value_series(
            tdf, port_panel, cash_start, fees, broker_name=broker
        )
        cash_now = float(cash_series.iloc[-1]) if not cash_series.empty else float(cash_start)

    nav = float(sum(p.market_value for p in positions.values()) + cash_now)
    bench = panel.get("VNINDEX", pd.Series(dtype=float)).dropna()
    risk_base = (
        portfolio_risk_summary(port_val, bench)
        if not port_val.empty
        else {"volatility": 0.0, "max_drawdown": 0.0, "beta": 0.0, "var_95": 0.0}
    )

    return {
        "positions": positions,
        "panel": panel,
        "cash": cash_now,
        "nav": nav,
        "risk": {
            "vol": float(risk_base.get("volatility", 0.0)),
            "beta": float(risk_base.get("beta", 0.0)),
            "cvar": float(min(0.0, risk_base.get("var_95", 0.0) * 1.15)),
            "mdd": float(risk_base.get("max_drawdown", 0.0)),
            "var_hist": float(risk_base.get("var_95", 0.0)),
        },
    }


def _capacity(positions: dict[str, Any], nav: float, tick_df: pd.DataFrame) -> dict[str, Any]:
    adtv_map = (
        tick_df.set_index("symbol").get("adtv_20d", pd.Series(dtype=float)).to_dict()
        if not tick_df.empty
        else {}
    )
    by_symbol: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for sym, p in positions.items():
        adtv = float(adtv_map.get(sym, 0.0) or 0.0)
        capacity_value = float(adtv * 0.05 * 3.0)
        limit_value = float(min(nav * 0.10, capacity_value))
        pos_value = float(p.market_value)
        breached = bool(pos_value > limit_value) if limit_value > 0 else False
        row = {
            "symbol": sym,
            "adtv_20d": adtv,
            "capacity_value": capacity_value,
            "limit_value": limit_value,
            "position_value": pos_value,
            "breached": breached,
        }
        by_symbol.append(row)
        if breached:
            flags.append(
                {
                    "symbol": sym,
                    "type": "capacity_breach",
                    "position_value": pos_value,
                    "limit_value": limit_value,
                }
            )
    return {"by_symbol": by_symbol, "flags": flags}


def _constraints_snapshot(db: Session) -> dict[str, Any]:
    rows = db.exec(
        select(RebalanceConstraintReport).order_by(
            RebalanceConstraintReport.as_of_date.desc(), RebalanceConstraintReport.id.desc()
        )
    ).all()
    if not rows:
        return {"active": [], "violations_pre": [], "violations_post": [], "distance_metrics": {}}
    r = rows[0].report_json or {}
    return {
        "active": r.get(
            "active_constraints",
            [
                "max_single",
                "max_sector",
                "min_cash",
                "liquidity cap",
                "cluster cap",
                "turnover cap",
            ],
        ),
        "violations_pre": r.get("violations_pre", []),
        "violations_post": r.get("violations_post", []),
        "distance_metrics": r.get("distance_metrics", {}),
    }


def build_portfolio_dashboard(db: Session, portfolio_id: int) -> dict[str, Any]:
    state = _load_portfolio_state(db, portfolio_id)
    tickers = db.exec(select(Ticker)).all()
    tick_df = (
        pd.DataFrame([t.model_dump() for t in tickers])
        if tickers
        else pd.DataFrame(columns=["symbol", "sector", "beta", "adtv_20d"])
    )

    positions = state["positions"]
    nav = float(state["nav"])
    cash = float(state["cash"])
    exposures_sector = exposure_by_sector(positions, tick_df) if positions else {}

    style_exposure = {
        "large_cap": float(
            sum(1 for _, p in positions.items() if p.market_value >= nav * 0.03)
            / max(1, len(positions))
        ),
        "mid_small": float(
            sum(1 for _, p in positions.items() if p.market_value < nav * 0.03)
            / max(1, len(positions))
        ),
    }

    sec_map = (
        tick_df.set_index("symbol").get("sector", pd.Series(dtype=object)).to_dict()
        if not tick_df.empty
        else {}
    )
    cluster_map = (
        tick_df.set_index("symbol").get("industry", pd.Series(dtype=object)).to_dict()
        if not tick_df.empty
        else {}
    )
    mv = pd.Series({s: p.market_value for s, p in positions.items()}, dtype=float)
    total_mv = float(mv.sum()) if not mv.empty else 0.0
    w = (mv / total_mv) if total_mv > 0 else pd.Series(dtype=float)

    names_contrib = [
        {"symbol": s, "contrib": float(abs(wt) * max(0.0, state["risk"]["vol"]))}
        for s, wt in w.sort_values(ascending=False).items()
    ]
    sectors_contrib = [
        {"sector": str(k), "contrib": float(v)}
        for k, v in w.groupby(w.index.map(lambda x: sec_map.get(x, "Unknown")))
        .sum()
        .sort_values(ascending=False)
        .items()
    ]
    clusters_contrib = [
        {"cluster": str(k), "contrib": float(v)}
        for k, v in w.groupby(w.index.map(lambda x: cluster_map.get(x, "Unknown")))
        .sum()
        .sort_values(ascending=False)
        .items()
    ]

    holdings = []
    for sym, p in positions.items():
        holdings.append(
            {
                "symbol": sym,
                "quantity": float(p.quantity),
                "price": float(p.market_price),
                "value": float(p.market_value),
                "weight": float(p.market_value / nav) if nav > 0 else 0.0,
                "sector": str(sec_map.get(sym, "Unknown")),
                "cluster": str(cluster_map.get(sym, "Unknown")),
            }
        )

    cap = _capacity(positions, nav, tick_df)
    constraints = _constraints_snapshot(db)

    as_of = max(state["panel"].index) if not state["panel"].empty else dt.date.today()
    return {
        "as_of_date": str(as_of),
        "portfolio_id": portfolio_id,
        "nav": nav,
        "cash": cash,
        "holdings": holdings,
        "exposures": {"sector": exposures_sector, "style": style_exposure},
        "risk": state["risk"],
        "risk_contrib": {
            "names": names_contrib,
            "sectors": sectors_contrib,
            "clusters": clusters_contrib,
        },
        "constraints": constraints,
        "capacity": cap,
    }


def build_rebalance_preview(db: Session, portfolio_id: int) -> dict[str, Any]:
    dash = build_portfolio_dashboard(db, portfolio_id)
    settings = get_settings()
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    exec_assump = load_execution_assumptions(settings.EXECUTION_MODEL_PATH)
    tickers = db.exec(select(Ticker)).all()
    tick_df = (
        pd.DataFrame([t.model_dump() for t in tickers])
        if tickers
        else pd.DataFrame(columns=["symbol"])
    )

    # reconstruct positions for suggest_rebalance
    from core.portfolio.analytics import Position

    pos = {
        h["symbol"]: Position(
            symbol=h["symbol"],
            quantity=h["quantity"],
            avg_cost=h["price"],
            market_price=h["price"],
            market_value=h["value"],
            unrealized_pnl=0.0,
        )
        for h in dash["holdings"]
    }

    reb = suggest_rebalance(pos, tick_df, cash=float(dash["cash"]), rules=None)
    trades: list[dict[str, Any]] = []
    cost_comm = 0.0
    cost_slip = 0.0
    cost_tax = 0.0

    reason_map = {
        "max_single_name_weight": "MAX_SINGLE",
        "max_sector_weight": "MAX_SECTOR",
        "target_cash_weight": "MIN_CASH",
    }

    for s in reb.get("suggestions", []):
        sym = str(s.get("symbol"))
        px = next((h["price"] for h in dash["holdings"] if h["symbol"] == sym), 0.0)
        qty = float(s.get("quantity", 0.0))
        notional = float(max(0.0, qty * px))
        adtv = float(
            (tick_df.set_index("symbol").get("adtv_20d", pd.Series(dtype=float)).to_dict()).get(
                sym, 0.0
            )
            if not tick_df.empty
            else 0.0
        )
        comm = fees.commission(notional, settings.BROKER_NAME)
        tax = fees.sell_tax(notional) if str(s.get("action", "")).upper() == "SELL" else 0.0
        slip_bps = slippage_bps(notional, adtv=adtv, atr_pct=0.02, assumptions=exec_assump)
        slip = notional * slip_bps / 10000.0
        cost_comm += comm
        cost_tax += tax
        cost_slip += slip

        key = reason_map.get(str(s.get("reason")), "REBALANCE_TO_TARGET")
        if key not in FIXED_REASON_KEYS:
            key = "REBALANCE_TO_TARGET"

        trades.append(
            {
                "symbol": sym,
                "side": str(s.get("action", "SELL")),
                "qty": qty,
                "price": px,
                "notional": notional,
                "reason_key": key,
                "explain": {
                    "reason_key": key,
                    "inputs": {
                        "weight": next(
                            (h["weight"] for h in dash["holdings"] if h["symbol"] == sym), 0.0
                        )
                    },
                },
            }
        )

    ac_schedule = []
    for t in trades:
        q = float(t["qty"])
        ac_schedule.append(
            {"symbol": t["symbol"], "day1": q * 0.5, "day2": q * 0.3, "day3": q * 0.2}
        )

    return {
        "trades": trades,
        "expected_costs": {
            "commission": float(cost_comm),
            "slippage": float(cost_slip),
            "sell_tax": float(cost_tax),
            "total": float(cost_comm + cost_slip + cost_tax),
        },
        "ac_schedule": ac_schedule,
        "explain_reason_keys": sorted(FIXED_REASON_KEYS),
    }


def apply_scenario_shocks(dashboard: dict[str, Any], preview: dict[str, Any]) -> dict[str, Any]:
    nav = float(dashboard.get("nav", 0.0))
    holdings = dashboard.get("holdings", [])
    sector_weights = dashboard.get("exposures", {}).get("sector", {})
    largest_sector = next(
        iter(sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)), (None, 0.0)
    )[0]
    largest_sector_weight = (
        float(sector_weights.get(largest_sector, 0.0)) if largest_sector else 0.0
    )

    s1_cost_total = float((preview.get("expected_costs") or {}).get("total", 0.0) * 2.0)
    s2_unfilled = float(sum(t.get("notional", 0.0) for t in preview.get("trades", [])) * 0.5)
    gross_exposure = float(sum(h.get("value", 0.0) for h in holdings))
    s3_nav = nav - gross_exposure * 0.05
    s4_nav = nav - gross_exposure * largest_sector_weight * 0.08

    return {
        "S1_cost_x2": {
            "delta_expected_cost": s1_cost_total
            - float((preview.get("expected_costs") or {}).get("total", 0.0))
        },
        "S2_fill_x0_5": {"unfilled_notional": s2_unfilled, "fill_ratio": 0.5},
        "S3_market_down_5": {"nav_after": s3_nav, "delta_nav": s3_nav - nav},
        "S4_largest_sector_down_8": {
            "largest_sector": largest_sector,
            "nav_after": s4_nav,
            "delta_nav": s4_nav - nav,
        },
    }


def resolve_portfolio_id(db: Session, portfolio_id: int | None) -> int:
    if portfolio_id is not None:
        return int(portfolio_id)
    p = db.exec(select(Portfolio).order_by(Portfolio.id.asc())).first()
    return int(p.id) if p else 0
