from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from api_fastapi.deps import get_db
from core.db.models import Portfolio, PriceOHLCV, Ticker, Trade
from core.execution_model import load_execution_assumptions, slippage_bps
from core.fees_taxes import FeesTaxes
from core.portfolio.analytics import (
    brinson_attribution_mvp,
    compute_portfolio_value_series,
    compute_positions_avg_cost,
    concentration_metrics,
    correlation_of_holdings,
    exposure_by_sector,
    infer_start_cash,
    portfolio_risk_summary,
    realized_pnl_breakdown,
    suggest_rebalance,
    time_weighted_return,
)
from core.regime import classify_market_regime
from core.settings import get_settings

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class CreatePortfolioRequest(BaseModel):
    name: str


class TradeIn(BaseModel):
    trade_date: str
    symbol: str
    side: str
    quantity: float
    price: float
    strategy_tag: str = ""
    notes: str = ""


class RebalanceRules(BaseModel):
    target_cash_weight: float = 0.10
    max_single_name_weight: float = 0.25
    max_sector_weight: float = 0.35


@router.get("", response_model=list[Portfolio])
def list_portfolios(db: Session = Depends(get_db)) -> List[Portfolio]:
    return list(db.exec(select(Portfolio)).all())


@router.post("", response_model=Portfolio)
def create_portfolio(payload: CreatePortfolioRequest, db: Session = Depends(get_db)) -> Portfolio:
    p = Portfolio(name=payload.name)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.post("/{portfolio_id}/trades/import")
def import_trades(portfolio_id: int, trades: List[TradeIn], db: Session = Depends(get_db)) -> Dict[str, Any]:
    settings = get_settings()
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    broker = settings.BROKER_NAME

    inserted = 0
    for t in trades:
        raw = f"{portfolio_id}|{t.trade_date}|{t.symbol}|{t.side}|{t.quantity}|{t.price}|{t.strategy_tag}|{t.notes}"
        ext = hashlib.sha1(raw.encode("utf-8")).hexdigest()

        if db.exec(select(Trade).where(Trade.external_id == ext)).first():
            continue

        # Store per-trade estimated commission/taxes (idempotent external_id)
        side = t.side.upper()
        qty = float(t.quantity)
        px = float(t.price)

        # For SELL, clamp at import time is not possible without positions state (do at analytics time).
        notional = qty * px
        commission = fees.commission(notional, broker)
        taxes = fees.sell_tax(notional) if side == "SELL" else 0.0

        tr = Trade(
            portfolio_id=portfolio_id,
            trade_date=pd.to_datetime(t.trade_date).date(),
            symbol=t.symbol,
            side=side,
            quantity=qty,
            price=px,
            strategy_tag=t.strategy_tag,
            notes=t.notes,
            commission=commission,
            taxes=taxes,
            external_id=ext,
        )
        db.add(tr)
        inserted += 1

    db.commit()
    return {"inserted": inserted}


@router.get("/{portfolio_id}/summary")
def portfolio_summary(portfolio_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    settings = get_settings()
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    exec_assump = load_execution_assumptions(settings.EXECUTION_MODEL_PATH)
    broker = settings.BROKER_NAME

    trades = db.exec(select(Trade).where(Trade.portfolio_id == portfolio_id)).all()
    if not trades:
        return {"portfolio_id": portfolio_id, "positions": [], "realized": [], "risk": {}, "twr": 0.0}

    tdf = pd.DataFrame([t.model_dump() for t in trades])

    # Prices panel (1D)
    prices = db.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()
    pdf = pd.DataFrame([p.model_dump() for p in prices])
    pdf["date"] = pd.to_datetime(pdf["timestamp"]).dt.date
    pdf = pdf.sort_values(["symbol", "date"])

    latest_prices = pdf.groupby("symbol")["close"].last().to_dict()

    # Positions + realized (Average Cost, fixed SELL clamp)
    positions, realized = compute_positions_avg_cost(tdf, latest_prices, fees, broker_name=broker)

    # Portfolio value series (infer non-negative start cash)
    symbols = sorted(set(list(positions.keys()) + ["VNINDEX"]))
    panel = (
        pdf[pdf["symbol"].isin(symbols)]
        .pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )

    bench = panel.get("VNINDEX", pd.Series(dtype=float)).dropna()
    port_panel = panel.drop(columns=["VNINDEX"], errors="ignore")

    cash_start = infer_start_cash(tdf, fees, broker_name=broker)
    port_val, cash_series = compute_portfolio_value_series(tdf, port_panel, cash_start, fees, broker_name=broker)
    twr = time_weighted_return(port_val)
    risk = portfolio_risk_summary(port_val, bench)

    # Exposure
    tickers = db.exec(select(Ticker)).all()
    tick_df = pd.DataFrame([t.model_dump() for t in tickers])
    exposure = exposure_by_sector(positions, tick_df)
    conc = concentration_metrics(positions)

    # Correlation matrix for holdings
    corr = correlation_of_holdings(port_panel, lookback=120)
    corr_payload = corr.round(4).to_dict() if not corr.empty else {}

    # Attribution (MVP, 20D sector returns)
    lookback = 20
    sym_ret_20d = (panel.tail(lookback).iloc[-1] / panel.tail(lookback).iloc[0] - 1.0).dropna() if panel.shape[0] >= lookback else pd.Series(dtype=float)

    sec_map = tick_df.set_index("symbol")["sector"].to_dict()
    pos_mv = pd.Series({s: p.market_value for s, p in positions.items()})
    total_mv = float(pos_mv.sum()) if not pos_mv.empty else 0.0
    w_port = (pos_mv / total_mv) if total_mv > 0 else pd.Series(dtype=float)
    w_port_sec = w_port.groupby(w_port.index.map(lambda s: sec_map.get(s, "Unknown"))).sum()

    port_sec_ret = sym_ret_20d.drop(labels=["VNINDEX"], errors="ignore").groupby(
        sym_ret_20d.drop(labels=["VNINDEX"], errors="ignore").index.map(lambda s: sec_map.get(s, "Unknown"))
    ).mean()

    mcap = tick_df.set_index("symbol")["market_cap"].astype(float).drop(labels=["VNINDEX"], errors="ignore")
    w_bench_sym = (mcap / mcap.sum()) if mcap.sum() > 0 else pd.Series(dtype=float)
    w_bench_sec = w_bench_sym.groupby(w_bench_sym.index.map(lambda s: sec_map.get(s, "Unknown"))).sum()
    bench_sec_ret = sym_ret_20d.drop(labels=["VNINDEX"], errors="ignore").groupby(
        sym_ret_20d.drop(labels=["VNINDEX"], errors="ignore").index.map(lambda s: sec_map.get(s, "Unknown"))
    ).mean()

    attribution = brinson_attribution_mvp(w_port_sec, port_sec_ret, w_bench_sec, bench_sec_ret) if not w_bench_sec.empty else {}

    # PnL breakdown
    realized_breakdown = realized_pnl_breakdown(realized)

    # Current cash (end)
    cash_now = float(cash_series.iloc[-1]) if not cash_series.empty else float(cash_start)

    # Sector PnL: unrealized by sector (MVP)
    unreal_by_sector: Dict[str, float] = {}
    for sym, pos in positions.items():
        sec = str(sec_map.get(sym, "Unknown"))
        unreal_by_sector[sec] = unreal_by_sector.get(sec, 0.0) + float(pos.unrealized_pnl)

    vn_close = panel.get("VNINDEX", pd.Series(dtype=float)).dropna()
    regime_series = classify_market_regime(vn_close)
    regime_state = str(regime_series.iloc[-1]) if not regime_series.empty else "sideway"

    # Rebalance suggestions (default rules + regime)
    rebalance = suggest_rebalance(positions, tick_df, cash=cash_now, rules=None, regime_state=regime_state)

    # assumptions panel payload
    sample_slippage_bps = slippage_bps(order_notional=1_000_000_000, adtv=20_000_000_000, atr_pct=0.02, assumptions=exec_assump)

    return {
        "portfolio_id": portfolio_id,
        "cash_start_inferred": cash_start,
        "cash_now": cash_now,
        "positions": [p.__dict__ for p in positions.values()],
        "realized_trades": realized.to_dict(orient="records") if not realized.empty else [],
        "realized_breakdown": realized_breakdown,
        "unrealized_by_sector": unreal_by_sector,
        "exposure_by_sector": exposure,
        "concentration": conc,
        "risk": risk,
        "twr": twr,
        "correlation_matrix": corr_payload,
        "attribution_mvp": {
            **attribution,
            "formula_note": "market=benchmark_return; selection=selection_effect; timing_proxy=allocation+interaction (MVP).",
        },
        "rebalance_mvp": rebalance,
        "assumptions": {
            "fees_taxes": {
                "commission_default": fees.default_commission_rate,
                "sell_tax_rate": fees.sell_tax_rate,
                "dividend_tax_rate": fees.dividend_tax_rate,
            },
            "execution": {
                "base_slippage_bps": exec_assump.base_slippage_bps,
                "k1_participation": exec_assump.k1_participation,
                "k2_volatility": exec_assump.k2_volatility,
                "limit_up_buy_fill_ratio": exec_assump.limit_up_buy_fill_ratio,
                "limit_down_sell_fill_ratio": exec_assump.limit_down_sell_fill_ratio,
                "sample_slippage_bps": sample_slippage_bps,
            },
            "regime": {"state": regime_state},
        },
        "notes": "Cost basis: Average Cost. Fees/taxes apply per configs/fees_taxes.yaml. Past performance does not guarantee future results; costs and fill assumptions materially affect outcomes.",
    }


@router.post("/{portfolio_id}/rebalance-suggest")
def rebalance_suggest(portfolio_id: int, payload: RebalanceRules, db: Session = Depends(get_db)) -> Dict[str, Any]:
    s = portfolio_summary(portfolio_id, db)
    if not s.get("positions"):
        return {"portfolio_id": portfolio_id, "suggestions": [], "note": "No positions"}
    settings = get_settings()
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    # reconstruct positions dict (minimal)
    from core.portfolio.analytics import Position  # local import

    pos = {p["symbol"]: Position(**p) for p in s["positions"]}
    tickers = db.exec(select(Ticker)).all()
    tick_df = pd.DataFrame([t.model_dump() for t in tickers])
    cash_now = float(s.get("cash_now", 0.0))
    rules = payload.model_dump()
    return suggest_rebalance(pos, tick_df, cash=cash_now, rules=rules)
