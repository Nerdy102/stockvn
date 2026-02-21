from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import yaml


OrderType = Literal["market", "limit"]
OrderSide = Literal["BUY", "SELL"]
TIF = Literal["IOC", "GTC"]


@dataclass
class OrderIntent:
    type: OrderType
    side: OrderSide
    qty: int
    limit_price: float | None = None
    tif: TIF = "IOC"


@dataclass
class Fill:
    ts: str
    side: OrderSide
    qty: int
    price: float
    fee: float
    tax: float
    slippage_cost: float
    status: Literal["FILLED", "PARTIAL", "UNFILLED"]
    remaining_qty: int = 0


@dataclass
class FundingRatePoint:
    ts: str
    symbol: str
    funding_rate: float


@dataclass
class FundingEvent:
    ts: str
    symbol: str
    funding_rate: float
    amount: float


@dataclass
class BacktestV3Config:
    market: Literal["vn", "crypto"] = "vn"
    trading_type: Literal["spot_paper", "perp_paper"] = "spot_paper"
    position_mode: Literal["long_only", "long_short"] = "long_only"
    order_type: OrderType = "market"
    tif: TIF = "IOC"
    execution: Literal["close", "next_bar"] = "close"
    target_notional_pct: float = 0.2
    participation_rate: float = 0.05
    initial_cash_vn: float = 100_000_000.0
    initial_cash_crypto: float = 10_000.0
    lot_size: int = 1
    engine_version: str = "v3"


@dataclass
class BacktestV3Report:
    net_return: float
    cagr: float
    mdd: float
    sharpe: float
    sortino: float
    calmar: float
    turnover: float
    win_rate: float
    profit_factor: float
    avg_hold_bars: float
    exposure_long_pct: float
    exposure_short_pct: float
    costs_breakdown: dict[str, float]
    config_hash: str
    dataset_hash: str
    code_hash: str
    report_id: str
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    funding_events: list[dict[str, Any]] = field(default_factory=list)


def _hash_obj(v: object) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _code_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown_code_hash"


def _dataset_hash(df: pd.DataFrame, symbol: str, timeframe: str, market: str) -> str:
    w = df.copy().reset_index(drop=True)
    if "timestamp" not in w.columns and "date" in w.columns:
        w["timestamp"] = w["date"]
    payload = w[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records")
    return _hash_obj({"market": market, "symbol": symbol, "timeframe": timeframe, "rows": len(w), "payload": payload})


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _desired_position(signal: str, mode: str) -> int:
    if mode == "long_only":
        if signal == "TANG":
            return 1
        if signal == "GIAM":
            return 0
        return 999
    if signal == "TANG":
        return 1
    if signal == "GIAM":
        return -1
    return 0


def _spread_bps(atr_pct: float) -> float:
    return max(2.0, min(30.0, 0.2 * atr_pct * 10000.0))


def _slippage_bps(order_notional: float, dollar_volume: float, atr_pct: float, model_cfg: dict[str, Any]) -> float:
    base_bps = float(model_cfg.get("base_bps", 3.0))
    k_atr = float(model_cfg.get("k_atr", 0.03))
    k_part = float(model_cfg.get("k_part", 50.0))
    return base_bps + k_atr * (atr_pct * 10000.0) + k_part * (order_notional / max(dollar_volume, 1.0))




def _load_corp_actions_cfg(path: str = "configs/corp_actions.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"enabled": False, "data_is_adjusted": "unknown", "apply_cash_dividend": True, "apply_split": True}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_corp_actions_for_symbol(symbol: str) -> list[dict[str, Any]]:
    base = Path("data_drop/corporate_actions")
    if not base.exists():
        return []
    out: list[dict[str, Any]] = []
    for fp in sorted(base.glob("*.csv")):
        try:
            rows = pd.read_csv(fp)
        except Exception:
            continue
        for _, r in rows.iterrows():
            if str(r.get("symbol", "")).upper() != symbol.upper():
                continue
            out.append({
                "symbol": str(r.get("symbol", "")).upper(),
                "ex_date": str(r.get("ex_date", ""))[:10],
                "action_type": str(r.get("action_type", "")).strip().lower(),
                "amount": float(r.get("amount", 0.0) or 0.0),
            })
    out.sort(key=lambda x: (x.get("ex_date", ""), x.get("action_type", "")))
    return out


def _bar_date_str(row: pd.Series) -> str:
    v = row.get("timestamp") or row.get("date")
    return str(v)[:10]

def _simulate_fill(order: OrderIntent, row: pd.Series, atr_pct: float, next_open: float | None, cfg: BacktestV3Config, fee_cfg: dict[str, Any], tax_rate: float) -> Fill:
    volume = float(row.get("volume", 0.0))
    max_fill_qty = int(math.floor(volume * cfg.participation_rate))
    fill_qty = min(order.qty, max(0, max_fill_qty))
    if fill_qty <= 0:
        return Fill(ts=str(row.get("timestamp") or row.get("date") or ""), side=order.side, qty=0, price=0.0, fee=0.0, tax=0.0, slippage_cost=0.0, status="UNFILLED", remaining_qty=order.qty)

    mid = (float(row["high"]) + float(row["low"])) / 2.0
    spread = _spread_bps(atr_pct)
    base_px = mid * (1 + spread / 20000.0) if order.side == "BUY" else mid * (1 - spread / 20000.0)

    if order.type == "limit":
        if order.side == "BUY":
            touched = float(row["low"]) <= float(order.limit_price or 0.0)
            if not touched:
                return Fill(ts=str(row.get("timestamp") or row.get("date") or ""), side=order.side, qty=0, price=0.0, fee=0.0, tax=0.0, slippage_cost=0.0, status="UNFILLED", remaining_qty=order.qty)
            base_px = min(float(order.limit_price or 0.0), float(next_open) if (cfg.execution == "next_bar" and next_open is not None) else float(order.limit_price or 0.0))
        else:
            touched = float(row["high"]) >= float(order.limit_price or 0.0)
            if not touched:
                return Fill(ts=str(row.get("timestamp") or row.get("date") or ""), side=order.side, qty=0, price=0.0, fee=0.0, tax=0.0, slippage_cost=0.0, status="UNFILLED", remaining_qty=order.qty)
            base_px = max(float(order.limit_price or 0.0), float(next_open) if (cfg.execution == "next_bar" and next_open is not None) else float(order.limit_price or 0.0))

    dollar_volume = float(row.get("close", 0.0) * row.get("volume", 0.0))
    order_notional = float(fill_qty * base_px)
    sbps = _slippage_bps(order_notional, dollar_volume, atr_pct, fee_cfg.get("slippage_model", {}))
    slipped_px = base_px * (1 + sbps / 10000.0) if order.side == "BUY" else base_px * (1 - sbps / 10000.0)

    if cfg.market == "vn":
        fee_rate = float(fee_cfg.get("commission_rate", 0.0))
    else:
        if cfg.trading_type == "perp_paper":
            fee_rate = float(fee_cfg.get("perp_taker_fee_rate", 0.0))
        else:
            fee_rate = float(fee_cfg.get("spot_taker_fee_rate", 0.0))
        if order.type == "limit":
            fee_rate = float(fee_cfg.get("maker_fee_rate", fee_rate))
    notional = float(fill_qty * slipped_px)
    fee = notional * fee_rate
    tax = notional * tax_rate if (cfg.market == "vn" and order.side == "SELL") else 0.0
    slippage_cost = notional * sbps / 10000.0
    status = "PARTIAL" if fill_qty < order.qty else "FILLED"
    return Fill(
        ts=str(row.get("timestamp") or row.get("date") or ""),
        side=order.side,
        qty=fill_qty,
        price=float(slipped_px),
        fee=float(fee),
        tax=float(tax),
        slippage_cost=float(slippage_cost),
        status=status,
        remaining_qty=max(0, int(order.qty - fill_qty)),
    )


def run_backtest_v3(
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: BacktestV3Config,
    signal_fn: Callable[[pd.DataFrame], str],
    fees_taxes_path: str,
    fees_crypto_path: str,
    execution_model_path: str,
    funding_rates: list[FundingRatePoint] | None = None,
    include_equity_curve: bool = False,
    include_trades: bool = False,
) -> BacktestV3Report:
    w = df.copy().reset_index(drop=True)
    if "timestamp" not in w.columns and "date" in w.columns:
        w["timestamp"] = pd.to_datetime(w["date"], errors="coerce")
    else:
        w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
    w = w.sort_values("timestamp").reset_index(drop=True)

    fees_vn = _load_yaml(fees_taxes_path)
    fees_crypto = _load_yaml(fees_crypto_path)
    exec_cfg = _load_yaml(execution_model_path)
    fee_cfg = fees_vn if config.market == "vn" else fees_crypto
    tax_rate = float(fees_vn.get("sell_tax_rate", 0.0))

    initial_cash = config.initial_cash_vn if config.market == "vn" else config.initial_cash_crypto
    cash = float(initial_cash)
    position = 0
    qty = 0
    entry_price = 0.0
    hold_bars = 0
    turnover = 0.0
    fee_total = 0.0
    tax_total = 0.0
    slippage_total = 0.0
    funding_total = 0.0
    wins = 0
    loss = 0
    gross_profit = 0.0
    gross_loss = 0.0
    long_bars = 0
    short_bars = 0

    eq: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    funding_events: list[dict[str, Any]] = []
    navs: list[float] = [cash]

    funding_map = {str(x.ts): float(x.funding_rate) for x in (funding_rates or [])}
    ca_cfg = _load_corp_actions_cfg()
    ca_events = _load_corp_actions_for_symbol(symbol) if bool(ca_cfg.get("enabled", False)) else []

    for i in range(len(w)):
        row = w.iloc[i]
        row_date = _bar_date_str(row)
        if ca_events and position > 0 and qty > 0:
            for ev in [x for x in ca_events if x.get("ex_date") == row_date]:
                a_type = str(ev.get("action_type", "")).lower()
                amount = float(ev.get("amount", 0.0) or 0.0)
                data_adj = str(ca_cfg.get("data_is_adjusted", "unknown")).lower()
                if a_type == "cash_dividend" and bool(ca_cfg.get("apply_cash_dividend", True)):
                    cash += float(qty) * amount
                if a_type == "split" and bool(ca_cfg.get("apply_split", True)) and data_adj == "unadjusted":
                    if amount > 0:
                        qty = int(round(float(qty) * amount))
                        entry_price = float(entry_price) / amount
        hist = w.iloc[: i + 1].copy()
        signal = signal_fn(hist)
        desired = _desired_position(signal, config.position_mode)
        if desired == 999:
            desired = position
        close_px = float(row["close"])
        atr_pct = float((float(row["high"]) - float(row["low"])) / max(close_px, 1e-9))
        nav = cash + (qty * close_px * position if position != 0 else 0.0)
        target_notional = nav * config.target_notional_pct
        lot = max(1, int(config.lot_size))
        target_qty = int(math.floor((target_notional / max(close_px, 1e-9)) / lot) * lot)

        action: OrderIntent | None = None
        if desired != position:
            if desired == 0 and position != 0:
                action = OrderIntent(type=config.order_type, side="SELL" if position > 0 else "BUY", qty=abs(qty), tif=config.tif, limit_price=close_px)
            elif position == 0 and desired != 0:
                action = OrderIntent(type=config.order_type, side="BUY" if desired > 0 else "SELL", qty=max(target_qty, lot), tif=config.tif, limit_price=close_px)
            elif desired != 0 and position != 0 and desired != position:
                # flip: close then open
                close_order = OrderIntent(type="market", side="SELL" if position > 0 else "BUY", qty=abs(qty), tif="IOC", limit_price=close_px)
                fill0 = _simulate_fill(close_order, row, atr_pct, float(w.iloc[i + 1]["open"]) if i + 1 < len(w) else None, config, {**fee_cfg, "slippage_model": exec_cfg}, tax_rate)
                if fill0.qty > 0:
                    notional0 = fill0.qty * fill0.price
                    if position > 0:
                        pnl = (fill0.price - entry_price) * fill0.qty - fill0.fee - fill0.tax - fill0.slippage_cost
                        cash += notional0 - fill0.fee - fill0.tax
                    else:
                        pnl = (entry_price - fill0.price) * fill0.qty - fill0.fee - fill0.slippage_cost
                        cash -= notional0 + fill0.fee
                    fee_total += fill0.fee
                    tax_total += fill0.tax
                    slippage_total += fill0.slippage_cost
                    gross_profit += max(pnl, 0.0)
                    gross_loss += abs(min(pnl, 0.0))
                    wins += 1 if pnl > 0 else 0
                    loss += 1 if pnl <= 0 else 0
                position = 0
                qty = 0
                action = OrderIntent(type=config.order_type, side="BUY" if desired > 0 else "SELL", qty=max(target_qty, lot), tif=config.tif, limit_price=close_px)

        if action is not None:
            next_open = float(w.iloc[i + 1]["open"]) if i + 1 < len(w) else None
            fill = _simulate_fill(action, row, atr_pct, next_open, config, {**fee_cfg, "slippage_model": exec_cfg}, tax_rate)
            if fill.qty > 0:
                notional = fill.qty * fill.price
                turnover += notional / max(initial_cash, 1e-9)
                fee_total += fill.fee
                tax_total += fill.tax
                slippage_total += fill.slippage_cost
                if action.side == "BUY":
                    if position == 0:
                        cash -= notional + fill.fee
                        position = 1
                        qty = fill.qty
                        entry_price = fill.price
                        hold_bars = 0
                    else:  # close short
                        pnl = (entry_price - fill.price) * fill.qty - fill.fee - fill.slippage_cost
                        cash -= notional + fill.fee
                        gross_profit += max(pnl, 0.0)
                        gross_loss += abs(min(pnl, 0.0))
                        wins += 1 if pnl > 0 else 0
                        loss += 1 if pnl <= 0 else 0
                        trades.append({"side": "SHORT", "entry_price": entry_price, "exit_price": fill.price, "qty": fill.qty, "pnl_net": pnl, "bars": hold_bars})
                        position = 0
                        qty = 0
                else:  # SELL
                    if position == 0 and config.position_mode == "long_short":
                        cash += notional - fill.fee
                        position = -1
                        qty = fill.qty
                        entry_price = fill.price
                        hold_bars = 0
                    else:
                        pnl = (fill.price - entry_price) * fill.qty - fill.fee - fill.tax - fill.slippage_cost
                        cash += notional - fill.fee - fill.tax
                        gross_profit += max(pnl, 0.0)
                        gross_loss += abs(min(pnl, 0.0))
                        wins += 1 if pnl > 0 else 0
                        loss += 1 if pnl <= 0 else 0
                        trades.append({"side": "LONG", "entry_price": entry_price, "exit_price": fill.price, "qty": fill.qty, "pnl_net": pnl, "bars": hold_bars})
                        position = 0
                        qty = 0

        hold_bars += 1 if position != 0 else 0
        long_bars += 1 if position > 0 else 0
        short_bars += 1 if position < 0 else 0

        # funding
        if config.market == "crypto" and config.trading_type == "perp_paper" and position != 0:
            f_rate = funding_map.get(str(row["timestamp"]), 0.0)
            if f_rate != 0.0:
                notional_pos = abs(qty * close_px)
                payment = notional_pos * f_rate
                amount = -payment if position > 0 else payment
                cash += amount
                funding_total += amount
                funding_events.append({"ts": str(row["timestamp"]), "symbol": symbol, "funding_rate": f_rate, "amount": amount})

        nav_now = cash + (qty * close_px * position if position != 0 else 0.0)
        navs.append(nav_now)
        peak = max(navs)
        dd = nav_now / max(peak, 1e-9) - 1.0
        eq.append({"time": str(row["timestamp"]), "nav": nav_now, "drawdown": dd})

    rets = pd.Series(navs).pct_change().dropna()
    net_return = navs[-1] / max(navs[0], 1e-9) - 1.0
    cagr = (1 + net_return) ** (252 / max(len(w), 1)) - 1 if len(w) > 1 else 0.0
    sharpe = float((rets.mean() / max(rets.std(ddof=0), 1e-9)) * math.sqrt(252)) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    sortino = float((rets.mean() / max(downside.std(ddof=0), 1e-9)) * math.sqrt(252)) if len(downside) > 1 else 0.0
    mdd = min([x["drawdown"] for x in eq], default=0.0)
    calmar = cagr / max(abs(mdd), 1e-9) if mdd < 0 else 0.0
    total_closed = max(1, wins + loss)
    win_rate = wins / total_closed
    profit_factor = gross_profit / max(gross_loss, 1e-9)
    avg_hold_bars = float(sum(t.get("bars", 0) for t in trades) / max(len(trades), 1))
    exposure_long = long_bars / max(len(w), 1)
    exposure_short = short_bars / max(len(w), 1)

    config_json = {
        "market": config.market,
        "trading_type": config.trading_type,
        "position_mode": config.position_mode,
        "order_type": config.order_type,
        "execution": config.execution,
        "target_notional_pct": config.target_notional_pct,
        "participation_rate": config.participation_rate,
    }
    config_hash = _hash_obj(config_json)
    dataset_hash = _dataset_hash(w, symbol, timeframe, config.market)
    code_hash = _code_hash()
    report_id = _hash_obj(config_json)  # temp
    report_id = _hash_obj({"config_json": config_json, "dataset_hash": dataset_hash, "code_hash": code_hash})

    return BacktestV3Report(
        net_return=float(net_return),
        cagr=float(cagr),
        mdd=float(mdd),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        turnover=float(turnover),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        avg_hold_bars=float(avg_hold_bars),
        exposure_long_pct=float(exposure_long),
        exposure_short_pct=float(exposure_short),
        costs_breakdown={
            "fee_total": float(fee_total),
            "tax_total": float(tax_total),
            "slippage_total": float(slippage_total),
            "funding_total": float(funding_total),
        },
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        code_hash=code_hash,
        report_id=report_id,
        equity_curve=eq if include_equity_curve else [],
        trades=trades if include_trades else [],
        funding_events=funding_events,
    )
