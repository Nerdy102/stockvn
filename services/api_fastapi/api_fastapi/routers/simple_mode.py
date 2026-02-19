from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from core.db.models import Fill, Portfolio, Trade
from core.execution_model import apply_slippage, load_execution_assumptions, slippage_bps
from core.fees_taxes import FeesTaxes
from core.market_rules import MarketRules
from core.settings import get_settings
from core.simple_mode.backtest import quick_backtest
from core.simple_mode.models import MODEL_PROFILES, run_signal
from core.simple_mode.orchestrator import build_client_order_id, generate_order_draft
from core.simple_mode.safety import ensure_disclaimers
from core.simple_mode.schemas import ConfirmExecutePayload
from core.universe.manager import UniverseManager
from data.providers.factory import get_provider
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from api_fastapi.deps import get_db

router = APIRouter(prefix="/simple", tags=["simple_mode"])

MAX_POINTS_PER_CHART = 300
MAX_SIGNAL_ROWS = 20
MAX_UNIVERSE_SCAN = 50
CACHE_TTL_SECONDS = 300


class RunSignalIn(BaseModel):
    symbol: str
    timeframe: str = "1D"
    model_id: str = "model_1"
    mode: str = "paper"
    market: str = Field(default="vn", pattern="^(vn|crypto)$")
    trading_type: str | None = Field(default=None, pattern="^(spot_paper|perp_paper)$")
    exchange: str | None = "binance_public"


class RunCompareIn(BaseModel):
    symbols: list[str] = Field(default_factory=list, min_length=1, max_length=20)
    timeframe: str = "1D"
    lookback_days: int = Field(default=252, ge=60, le=756)
    detail_level: str = Field(default="tóm tắt", pattern="^(tóm tắt|chi tiết)$")
    include_equity_curve: bool = False
    include_trades: bool = False
    execution: str = Field(
        default="giá đóng cửa (close)",
        pattern=r"^(giá đóng cửa \(close\)|thanh nến kế tiếp \(next-bar\))$",
    )
    market: str = Field(default="vn", pattern="^(vn|crypto)$")
    trading_type: str | None = Field(default=None, pattern="^(spot_paper|perp_paper)$")
    exchange: str | None = "binance_public"


def _resolve_market_mode(market: str, trading_type: str | None) -> tuple[str, str, str]:
    mk = (market or "vn").strip().lower()
    tt = (trading_type or "spot_paper").strip().lower()
    if mk == "vn":
        return "vn", "vn_long_only", "spot_paper"
    if tt == "perp_paper":
        return "crypto", "long_short", "perp_paper"
    return "crypto", "long_only", "spot_paper"


def _map_signal_side(signal: Any, position_mode: str) -> Any:
    if signal.proposed_side == "SELL" and position_mode == "long_short":
        signal.proposed_side = "SHORT"
    return signal


def _build_provider(settings: Any, market: str, exchange: str | None = None) -> Any:
    if market == "crypto":
        from data.providers.crypto_public_ohlcv import CryptoPublicOHLCVProvider

        return CryptoPublicOHLCVProvider(exchange=exchange or settings.CRYPTO_DEFAULT_EXCHANGE)
    return get_provider(settings)


def _hash_obj(v: object) -> str:
    s = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _write_simple_audit(event: dict[str, Any]) -> str:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    audit_id = f"simple-audit-{ts}"
    out_dir = Path("artifacts/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "simple_mode_audit.jsonl"
    payload = {"audit_id": audit_id, **event}
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return audit_id


def _cache_path(key: str) -> Path:
    out_dir = Path("artifacts/cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"simple_dashboard_{key}.json"


def _read_dashboard_cache(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expires_at = dt.datetime.fromisoformat(str(data.get("expires_at")))
        if dt.datetime.utcnow() > expires_at:
            return None
        return data.get("payload")
    except Exception:
        return None


def _write_dashboard_cache(key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(key)
    data = {
        "expires_at": (dt.datetime.utcnow() + dt.timedelta(seconds=CACHE_TTL_SECONDS)).isoformat(),
        "payload": payload,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _compare_cache_path(key: str) -> Path:
    out_dir = Path("artifacts/cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"simple_compare_{key}.json"


def _read_compare_cache(key: str) -> dict[str, Any] | None:
    path = _compare_cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expires_at = dt.datetime.fromisoformat(str(data.get("expires_at")))
        if dt.datetime.utcnow() > expires_at:
            return None
        return data.get("payload")
    except Exception:
        return None


def _write_compare_cache(key: str, payload: dict[str, Any]) -> None:
    path = _compare_cache_path(key)
    data = {
        "expires_at": (dt.datetime.utcnow() + dt.timedelta(seconds=CACHE_TTL_SECONDS)).isoformat(),
        "payload": payload,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _as_of_date_for_symbols(provider: Any, symbols: list[str], timeframe: str) -> str:
    latest: dt.date | None = None
    for symbol in symbols:
        df = provider.get_ohlcv(symbol, timeframe)
        if df.empty:
            continue
        val = df.iloc[-1].get("date") or df.iloc[-1].get("timestamp")
        if val is None:
            continue
        s = str(val)[:10]
        try:
            d = dt.date.fromisoformat(s)
            if latest is None or d > latest:
                latest = d
        except ValueError:
            continue
    return (latest or dt.date.today()).isoformat()


def _resolve_universe_symbols(
    db: Session, universe: str, limit: int = MAX_UNIVERSE_SCAN
) -> list[str]:
    manager = UniverseManager(db)
    as_of = dt.date.today()
    symbols, _ = manager.universe(date=as_of, name=universe)
    out = sorted(set(symbols))[:limit]
    if not out:
        return ["FPT", "VNM", "VCB", "MWG", "HPG"]
    return out


def _market_today_summary(provider: Any, symbols: list[str], timeframe: str) -> dict[str, Any]:
    bench_df = provider.get_ohlcv("VNINDEX", timeframe)
    benchmark = "VNINDEX"
    benchmark_note = "Dùng VNINDEX làm chỉ số tham chiếu chính."
    if bench_df.empty or len(bench_df) < 2:
        benchmark = symbols[0] if symbols else "VN30"
        bench_df = provider.get_ohlcv(benchmark, timeframe)
        benchmark_note = f"Thiếu VNINDEX, dùng mã đại diện {benchmark} làm tham chiếu."
    if bench_df.empty or len(bench_df) < 2:
        return {
            "text": "Thiếu dữ liệu để tóm tắt thị trường hôm nay. Hãy bấm Đồng bộ dữ liệu (Sync data) hoặc dùng demo.",
            "benchmark": benchmark,
            "benchmark_note": benchmark_note,
            "daily_change_pct": 0.0,
            "breadth_up": 0,
            "breadth_down": 0,
            "breadth_flat": 0,
            "liquidity_vs_avg20": 0.0,
        }

    last_close = float(bench_df.iloc[-1]["close"])
    prev_close = float(bench_df.iloc[-2]["close"])
    daily_change_pct = (last_close / max(prev_close, 1e-9) - 1) * 100

    up = down = flat = 0
    total_liq = 0.0
    total_avg20 = 0.0
    for symbol in symbols[:MAX_UNIVERSE_SCAN]:
        df = provider.get_ohlcv(symbol, timeframe)
        if len(df) < 2:
            continue
        c1 = float(df.iloc[-1]["close"])
        c0 = float(df.iloc[-2]["close"])
        r = c1 - c0
        if r > 0:
            up += 1
        elif r < 0:
            down += 1
        else:
            flat += 1
        vol_now = float(df.iloc[-1].get("volume", 0.0))
        vol_avg20 = (
            float(df["volume"].tail(20).mean()) if len(df) >= 20 else float(df["volume"].mean())
        )
        total_liq += vol_now
        total_avg20 += vol_avg20

    liq_ratio = (total_liq / max(total_avg20, 1e-9)) if total_avg20 > 0 else 0.0
    text = (
        f"{benchmark_note} Chỉ số tham chiếu biến động {daily_change_pct:.2f}% trong ngày. "
        f"Độ rộng thị trường: {up} mã tăng, {down} mã giảm, {flat} mã đi ngang. "
        f"Thanh khoản so với trung bình 20 phiên: {liq_ratio:.2f} lần."
    )
    return {
        "text": text,
        "benchmark": benchmark,
        "benchmark_note": benchmark_note,
        "daily_change_pct": daily_change_pct,
        "breadth_up": up,
        "breadth_down": down,
        "breadth_flat": flat,
        "liquidity_vs_avg20": liq_ratio,
    }


def _scan_signals(
    provider: Any,
    symbols: list[str],
    timeframe: str,
    limit_signals: int,
    *,
    position_mode: str = "long_only",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_labels = {
        "model_1": "Mô hình 1 — Xu hướng (Trend-following)",
        "model_2": "Mô hình 2 — Hồi quy về trung bình (Mean-reversion)",
        "model_3": "Mô hình 3 — Kết hợp nhân tố + chế độ thị trường (Factor + Regime)",
    }
    buy_rows: list[dict[str, Any]] = []
    sell_rows: list[dict[str, Any]] = []
    for symbol in symbols[:MAX_UNIVERSE_SCAN]:
        df = provider.get_ohlcv(symbol, timeframe)
        if df.empty:
            continue
        for model_id in ["model_1", "model_2", "model_3"]:
            sig = _map_signal_side(run_signal(model_id, symbol, timeframe, df), position_mode)
            row = {
                "symbol": symbol,
                "model": model_labels[model_id],
                "model_id": model_id,
                "signal": sig.signal,
                "confidence": sig.confidence,
                "reason": " ".join(sig.explanation[:1]),
                "risks": sig.risks[:2],
                "open_simple_mode": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "model_id": model_id,
                },
            }
            if sig.proposed_side == "BUY":
                buy_rows.append(row)
            elif sig.proposed_side == "SELL":
                sell_rows.append(row)
    return buy_rows[:limit_signals], sell_rows[:limit_signals]


def _model_performance(
    provider: Any,
    symbols: list[str],
    timeframe: str,
    lookback_sessions: int,
    *,
    position_mode: str = "long_only",
) -> list[dict[str, Any]]:
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_sessions)
    rows: list[dict[str, Any]] = []
    for model_id in ["model_1", "model_2", "model_3"]:
        reports = []
        for symbol in symbols:
            df = provider.get_ohlcv(symbol, timeframe)
            reports.append(
                quick_backtest(model_id, symbol, df, start, end, position_mode=position_mode)
            )
        out = {
            "model_id": model_id,
            "net_return": sum(r.net_return for r in reports) / len(reports),
            "cagr": sum(r.cagr for r in reports) / len(reports),
            "mdd": sum(r.mdd for r in reports) / len(reports),
            "sharpe": sum(r.sharpe for r in reports) / len(reports),
            "turnover": sum(r.turnover for r in reports) / len(reports),
            "config_hash": reports[0].config_hash,
            "dataset_hash": reports[0].dataset_hash,
            "code_hash": reports[0].code_hash,
            "long_exposure": sum(r.long_exposure for r in reports) / len(reports),
            "short_exposure": sum(r.short_exposure for r in reports) / len(reports),
        }
        out["report_id"] = f"{out['config_hash']}-{out['dataset_hash']}-{out['code_hash']}"
        rows.append(out)
    rows.sort(key=lambda x: x["sharpe"], reverse=True)
    return rows


def _safe_date(v: Any) -> str:
    return str(v)[:10]


def _run_compare_v2(
    *,
    provider: Any,
    model_id: str,
    symbols: list[str],
    timeframe: str,
    lookback_days: int,
    execution_mode: str,
    include_equity_curve: bool,
    include_trades: bool,
    position_mode: str = "long_only",
) -> dict[str, Any]:
    settings = get_settings()
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    exec_assump = load_execution_assumptions(settings.EXECUTION_MODEL_PATH)
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)

    commission_rate = fees.commission_rate(settings.BROKER_NAME)
    lot_size = int(mr.quantity_rules.get("board_lot", 100))
    fixed_notional = 100_000_000.0

    nav = 1.0
    cash = 1.0
    pos_qty = 0
    pos_entry_price = 0.0
    pos_side = "FLAT"
    equity_curve: list[dict[str, Any]] = []
    trade_list: list[dict[str, Any]] = []
    turnover = 0.0

    benchmark = 1.0
    has_benchmark = False
    benchmark_df = provider.get_ohlcv("VNINDEX", timeframe)
    if not benchmark_df.empty and len(benchmark_df) >= 2:
        benchmark_df = benchmark_df.copy()
        benchmark_df["d"] = benchmark_df["date"].astype(str).str[:10]
        benchmark_df = benchmark_df[
            (benchmark_df["d"] >= start.isoformat()) & (benchmark_df["d"] <= end.isoformat())
        ]
        if len(benchmark_df) >= 2:
            has_benchmark = True

    combined_rows = 0
    for symbol in symbols:
        df = provider.get_ohlcv(symbol, timeframe)
        if df.empty:
            continue
        if "date" in df.columns:
            df = df.copy()
            df["d"] = df["date"].astype(str).str[:10]
            w = df[(df["d"] >= start.isoformat()) & (df["d"] <= end.isoformat())].copy()
        else:
            w = df.copy()
            w["d"] = w.index.astype(str).str[:10]
        if len(w) < 30:
            continue
        combined_rows += len(w)
        w = w.reset_index(drop=True)

        for i in range(20, len(w)):
            hist = w.iloc[: i + 1].copy()
            sig = _map_signal_side(run_signal(model_id, symbol, timeframe, hist), position_mode)
            today = w.iloc[i]
            trade_bar_idx = min(i + 1, len(w) - 1) if execution_mode.endswith("(next-bar)") else i
            trade_bar = w.iloc[trade_bar_idx]
            trade_px_raw = float(trade_bar["close"])
            trade_dt = _safe_date(trade_bar.get("date", trade_bar.get("d", "")))
            mark_px = float(today["close"])

            if sig.proposed_side == "BUY" and pos_qty == 0:
                qty_raw = fixed_notional / max(trade_px_raw, 1.0)
                qty = int(math.floor(qty_raw / max(lot_size, 1)) * lot_size)
                if qty <= 0:
                    continue
                slip_bps = slippage_bps(
                    float(qty * trade_px_raw),
                    max(float(today.get("volume", 0.0) * trade_px_raw), 1.0),
                    0.01,
                    exec_assump,
                )
                exec_px = mr.round_price(
                    apply_slippage(trade_px_raw, "BUY", slip_bps), direction="up"
                )
                notional = float(exec_px * qty)
                fee = fees.commission(notional, settings.BROKER_NAME)
                cash -= (notional + fee) / fixed_notional
                pos_qty = qty
                pos_entry_price = exec_px
                pos_side = "LONG"
                turnover += notional / fixed_notional
                trade_list.append(
                    {
                        "entry_date": trade_dt,
                        "exit_date": None,
                        "side": "BUY",
                        "qty": qty,
                        "entry_price": exec_px,
                        "exit_price": None,
                        "pnl_gross": 0.0,
                        "fee": fee,
                        "tax": 0.0,
                        "slippage_est": (exec_px - trade_px_raw) * qty,
                        "pnl_net": -fee,
                        "lot_size": lot_size,
                    }
                )

            if sig.proposed_side == "SHORT" and pos_qty == 0 and position_mode == "long_short":
                qty_raw = fixed_notional / max(trade_px_raw, 1.0)
                qty = int(math.floor(qty_raw / max(lot_size, 1)) * lot_size)
                if qty <= 0:
                    continue
                slip_bps = slippage_bps(
                    float(qty * trade_px_raw),
                    max(float(today.get("volume", 0.0) * trade_px_raw), 1.0),
                    0.01,
                    exec_assump,
                )
                exec_px = mr.round_price(
                    apply_slippage(trade_px_raw, "SELL", slip_bps), direction="down"
                )
                notional = float(exec_px * qty)
                fee = fees.commission(notional, settings.BROKER_NAME)
                cash += (notional - fee) / fixed_notional
                pos_qty = qty
                pos_entry_price = exec_px
                pos_side = "LONG"
                pos_side = "SHORT"
                turnover += notional / fixed_notional

            if sig.proposed_side == "BUY" and pos_qty > 0 and pos_side == "SHORT":
                slip_bps = slippage_bps(
                    float(pos_qty * trade_px_raw),
                    max(float(today.get("volume", 0.0) * trade_px_raw), 1.0),
                    0.01,
                    exec_assump,
                )
                exec_px = mr.round_price(
                    apply_slippage(trade_px_raw, "BUY", slip_bps), direction="up"
                )
                notional = float(exec_px * pos_qty)
                fee = fees.commission(notional, settings.BROKER_NAME)
                pnl_gross = (pos_entry_price - exec_px) * pos_qty
                cash -= (notional + fee) / fixed_notional
                turnover += notional / fixed_notional
                pos_qty = 0
                pos_entry_price = 0.0
                pos_side = "FLAT"
                pos_side = "FLAT"

            if sig.proposed_side == "SELL" and pos_qty > 0:
                slip_bps = slippage_bps(
                    float(pos_qty * trade_px_raw),
                    max(float(today.get("volume", 0.0) * trade_px_raw), 1.0),
                    0.01,
                    exec_assump,
                )
                exec_px = mr.round_price(
                    apply_slippage(trade_px_raw, "SELL", slip_bps), direction="down"
                )
                notional = float(exec_px * pos_qty)
                fee = fees.commission(notional, settings.BROKER_NAME)
                tax = fees.sell_tax(notional)
                pnl_gross = (exec_px - pos_entry_price) * pos_qty
                pnl_net = pnl_gross - fee - tax
                cash += (notional - fee - tax) / fixed_notional
                turnover += notional / fixed_notional
                if trade_list:
                    trade_list[-1].update(
                        {
                            "exit_date": trade_dt,
                            "exit_price": exec_px,
                            "pnl_gross": pnl_gross,
                            "fee": trade_list[-1]["fee"] + fee,
                            "tax": tax,
                            "slippage_est": trade_list[-1]["slippage_est"]
                            + (trade_px_raw - exec_px) * pos_qty,
                            "pnl_net": pnl_net,
                        }
                    )
                pos_qty = 0
                pos_entry_price = 0.0
                pos_side = "FLAT"

            signed_mv = (
                (pos_qty * mark_px) / fixed_notional
                if pos_side != "SHORT"
                else -(pos_qty * mark_px) / fixed_notional
            )
            nav = cash + signed_mv
            peak = max(nav, max([x["nav"] for x in equity_curve], default=nav))
            drawdown = (nav / max(peak, 1e-9)) - 1
            equity_curve.append(
                {
                    "date": _safe_date(today.get("date", today.get("d", ""))),
                    "nav": nav,
                    "drawdown": drawdown,
                }
            )

    if not equity_curve:
        equity_curve = [{"date": end.isoformat(), "nav": 1.0, "drawdown": 0.0}]
    nav_series = [float(x["nav"]) for x in equity_curve]
    rets = []
    for i in range(1, len(nav_series)):
        prev = nav_series[i - 1]
        rets.append((nav_series[i] / prev - 1.0) if prev > 0 else 0.0)
    mdd = min([float(x["drawdown"]) for x in equity_curve], default=0.0)
    net_return = nav_series[-1] - 1.0
    vol = (
        float(
            (sum((r - (sum(rets) / max(len(rets), 1))) ** 2 for r in rets) / max(len(rets), 1))
            ** 0.5
        )
        if rets
        else 0.0
    )
    sharpe = float((sum(rets) / max(len(rets), 1)) / vol * (252**0.5)) if vol > 0 else 0.0
    cagr = float((nav_series[-1]) ** (252 / max(len(nav_series), 1)) - 1)

    hash_payload = {
        "model_id": model_id,
        "symbols": symbols,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "execution": execution_mode,
        "rows": combined_rows,
    }
    config_hash = _hash_obj(
        {
            k: hash_payload[k]
            for k in ["model_id", "symbols", "timeframe", "lookback_days", "execution"]
        }
    )
    dataset_hash = _hash_obj({"rows": combined_rows, "symbols": symbols})
    code_hash = "simple_mode_backtest_v2"
    report_id = _hash_obj(
        {"config_hash": config_hash, "dataset_hash": dataset_hash, "code_hash": code_hash}
    )

    out = {
        "model_id": model_id,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": sharpe,
        "turnover": turnover,
        "net_return_after_fees_taxes": net_return,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "code_hash": code_hash,
        "report_id": report_id,
        "benchmark": {
            "status": "ok" if has_benchmark else "không có benchmark",
            "net_return": (
                float(benchmark_df.iloc[-1]["close"] / benchmark_df.iloc[0]["close"] - 1)
                if has_benchmark
                else None
            ),
        },
    }
    if include_equity_curve:
        out["equity_curve"] = equity_curve[:MAX_POINTS_PER_CHART]
    if include_trades:
        out["trade_list"] = trade_list
    return out


def _paper_portfolio_summary(db: Session, provider: Any, timeframe: str = "1D") -> dict[str, Any]:
    trades = db.exec(select(Trade).where(Trade.strategy_tag.like("simple:%"))).all()
    if not trades:
        return {
            "status": "empty",
            "message": "Chưa có giao dịch giấy — hãy tạo lệnh nháp và xác nhận.",
            "total_orders": 0,
            "open_positions": 0,
            "cash_ratio": 1.0,
            "pnl": 0.0,
            "top_positions": [],
        }

    pos: dict[str, dict[str, float]] = {}
    for t in trades:
        q = float(t.quantity)
        side = str(t.side).upper()
        signed = q if side == "BUY" else -q
        if t.symbol not in pos:
            pos[t.symbol] = {"qty": 0.0, "cost": 0.0}
        pos[t.symbol]["qty"] += signed
        pos[t.symbol]["cost"] += signed * float(t.price)

    top_positions: list[dict[str, Any]] = []
    total_mv = 0.0
    total_cost = 0.0
    for symbol, p in pos.items():
        qty = p["qty"]
        if qty == 0:
            continue
        df = provider.get_ohlcv(symbol, timeframe)
        last = float(df.iloc[-1]["close"]) if not df.empty else 0.0
        mv = qty * last
        cost = p["cost"]
        pnl = mv - cost
        total_mv += mv
        total_cost += cost
        avg_cost = (cost / qty) if qty != 0 else 0.0
        top_positions.append(
            {
                "symbol": symbol,
                "quantity": qty,
                "avg_cost": avg_cost,
                "market_value": mv,
                "pnl": pnl,
            }
        )
    top_positions.sort(key=lambda x: abs(x["market_value"]), reverse=True)
    cash_ratio = 0.0 if total_mv > 0 else 1.0
    return {
        "status": "ok",
        "message": "Tóm tắt danh mục giao dịch giấy (Paper portfolio).",
        "total_orders": len(trades),
        "open_positions": len([p for p in top_positions if p["quantity"] != 0]),
        "cash_ratio": cash_ratio,
        "pnl": total_mv - total_cost,
        "top_positions": top_positions[:5],
    }


def _data_status(provider: Any, symbols: list[str], timeframe: str) -> dict[str, Any]:
    rows = 0
    last_update = None
    for symbol in symbols[:MAX_UNIVERSE_SCAN]:
        df = provider.get_ohlcv(symbol, timeframe)
        rows += len(df)
        if not df.empty:
            last_update = str(df.iloc[-1].get("date") or df.iloc[-1].get("timestamp"))
    return {
        "provider": str(type(provider).__name__),
        "symbols_count": len(symbols),
        "rows": rows,
        "timeframes": ["1D", "60m"],
        "last_update": last_update,
    }


@router.get("/models")
def get_models() -> dict[str, Any]:
    settings = get_settings()
    return {
        "models": [m.__dict__ for m in MODEL_PROFILES],
        "live_enabled": str(__import__("os").getenv("ENABLE_LIVE_TRADING", "false")).lower()
        == "true",
        "max_points_per_chart": MAX_POINTS_PER_CHART,
        "warning": "Tín hiệu nghiên cứu (Research signal), không phải lời khuyên đầu tư (Not investment advice).",
        "provider": settings.DATA_PROVIDER,
    }


@router.get("/dashboard")
def get_dashboard(
    universe: str = Query(default="VN30", pattern="^(ALL|VN30|VNINDEX)$"),
    timeframe: str = Query(default="1D", pattern="^(1D|60m)$"),
    limit_signals: int = Query(default=10, ge=1, le=20),
    lookback_sessions: int = Query(default=252, ge=60, le=756),
    market: str = Query(default="vn", pattern="^(vn|crypto|both)$"),
    trading_type: str = Query(default="spot_paper", pattern="^(spot_paper|perp_paper)$"),
    exchange: str = Query(default="binance_public"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    settings = get_settings()
    mk = market if market != "both" else "crypto"
    _, position_mode, normalized_trading_type = _resolve_market_mode(mk, trading_type)
    provider = _build_provider(settings, mk, exchange)
    symbols = (
        _resolve_universe_symbols(db=db, universe=universe, limit=MAX_UNIVERSE_SCAN)
        if mk == "vn"
        else ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    )
    as_of_date = _as_of_date_for_symbols(provider, symbols, timeframe)

    cache_key = f"{universe}-{timeframe}-{limit_signals}-{lookback_sessions}-{as_of_date}".replace(
        "/", "_"
    )
    cached = _read_dashboard_cache(cache_key)
    if cached is not None:
        return cached

    buy_rows, sell_rows = _scan_signals(
        provider, symbols, timeframe, limit_signals, position_mode=position_mode
    )
    leaderboard = _model_performance(
        provider,
        symbols[: max(1, min(20, len(symbols)))],
        timeframe,
        lookback_sessions,
        position_mode=position_mode,
    )
    payload = {
        "as_of_date": as_of_date,
        "market": mk,
        "trading_type": normalized_trading_type,
        "market_summary": _market_today_summary(provider, symbols, timeframe),
        "market_today_summary": _market_today_summary(provider, symbols, timeframe),
        "buy_candidates": buy_rows,
        "sell_candidates": sell_rows,
        "signals_buy_candidates": buy_rows,
        "signals_sell_candidates": sell_rows,
        "model_leaderboard": leaderboard,
        "model_performance_leaderboard": leaderboard,
        "paper_portfolio_summary": _paper_portfolio_summary(db, provider, timeframe),
        "data_status": _data_status(provider, symbols, timeframe),
        "disclaimers": [
            "Đây là tín hiệu nghiên cứu (Research signal), không phải lời khuyên đầu tư (Not investment advice).",
            "Quá khứ không đảm bảo tương lai (Past performance is not indicative of future results).",
            "Có thể thua lỗ (Risk of loss).",
            "Nếu dưới 18 tuổi (Under 18), có thể cần người giám hộ/đủ tuổi để mở tài khoản; không hỗ trợ lách luật.",
        ],
    }
    _write_dashboard_cache(cache_key, payload)

    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_id = leaderboard[0]["report_id"] if leaderboard else cache_key
    report_path = report_dir / f"simple_dashboard_{report_id}.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    payload["report_id"] = report_id
    return payload


@router.get("/dashboard/report/{report_id}")
def get_dashboard_report(report_id: str) -> dict[str, Any]:
    path = Path("artifacts/reports") / f"simple_dashboard_{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/dashboard/refresh")
def refresh_dashboard() -> dict[str, Any]:
    return {
        "status": "ok",
        "message": "Đã làm mới dữ liệu dashboard (offline demo: refresh nhẹ).",
    }


@router.post("/run_signal")
def run_signal_api(payload: RunSignalIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    del db
    settings = get_settings()
    mk, position_mode, normalized_trading_type = _resolve_market_mode(
        payload.market, payload.trading_type
    )
    provider = _build_provider(settings, mk, payload.exchange)
    df = provider.get_ohlcv(payload.symbol, payload.timeframe)
    if len(df) > MAX_POINTS_PER_CHART:
        step = max(1, len(df) // MAX_POINTS_PER_CHART)
        df = df.iloc[::step].copy()

    signal = _map_signal_side(
        run_signal(payload.model_id, payload.symbol, payload.timeframe, df), position_mode
    )
    if not df.empty:
        df = df.copy()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    draft = generate_order_draft(
        signal=signal,
        market_rules=mr,
        fees_taxes=fees,
        mode=payload.mode,
        allow_short=(mk == "crypto" and normalized_trading_type == "perp_paper"),
        has_open_position=(mk == "crypto" and normalized_trading_type == "perp_paper"),
    )

    chart = []
    for _, row in df.tail(MAX_POINTS_PER_CHART).iterrows():
        chart.append(
            {
                "time": str(row.get("date") or row.get("timestamp") or ""),
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", 0.0)),
                "volume": float(row.get("volume", 0.0)),
                "ema20": float(row.get("ema20", row.get("close", 0.0))),
                "ema50": float(row.get("ema50", row.get("close", 0.0))),
            }
        )

    data_status = {
        "rows": int(len(df)),
        "last_update": str(df.iloc[-1].get("date") if not df.empty else ""),
    }
    return {
        "market": mk,
        "trading_type": normalized_trading_type,
        "signal": signal.model_dump(),
        "draft": None if draft is None else draft.model_dump(),
        "data_status": data_status,
        "chart": chart,
    }


@router.post("/run_compare")
def run_compare_api(payload: RunCompareIn) -> dict[str, Any]:
    settings = get_settings()
    mk, position_mode, normalized_trading_type = _resolve_market_mode(
        payload.market, payload.trading_type
    )
    provider = _build_provider(settings, mk, payload.exchange)
    end = dt.date.today()
    start = end - dt.timedelta(days=payload.lookback_days)

    is_detail = payload.detail_level == "chi tiết"
    include_equity_curve = payload.include_equity_curve or is_detail
    include_trades = payload.include_trades or is_detail
    as_of_date = _as_of_date_for_symbols(provider, payload.symbols, payload.timeframe)
    compare_cache_key = _hash_obj(
        {
            "as_of_date": as_of_date,
            "market": mk,
            "trading_type": normalized_trading_type,
            "symbols": payload.symbols,
            "timeframe": payload.timeframe,
            "lookback_days": payload.lookback_days,
            "detail_level": payload.detail_level,
            "execution": payload.execution,
            "include_equity_curve": include_equity_curve,
            "include_trades": include_trades,
            "market": mk,
            "trading_type": normalized_trading_type,
        }
    )
    cached_compare = _read_compare_cache(compare_cache_key)
    if cached_compare is not None:
        return cached_compare

    rows: list[dict[str, Any]] = []
    for model in ["model_1", "model_2", "model_3"]:
        if is_detail:
            avg = _run_compare_v2(
                provider=provider,
                model_id=model,
                symbols=payload.symbols,
                timeframe=payload.timeframe,
                lookback_days=payload.lookback_days,
                execution_mode=payload.execution,
                include_equity_curve=include_equity_curve,
                include_trades=include_trades,
                position_mode=position_mode,
            )
        else:
            reports = []
            for symbol in payload.symbols:
                df = provider.get_ohlcv(symbol, payload.timeframe)
                reports.append(
                    quick_backtest(model, symbol, df, start, end, position_mode=position_mode)
                )
            avg = {
                "model_id": model,
                "cagr": sum(r.cagr for r in reports) / len(reports),
                "mdd": sum(r.mdd for r in reports) / len(reports),
                "sharpe": sum(r.sharpe for r in reports) / len(reports),
                "sortino": sum(r.sortino for r in reports) / len(reports),
                "turnover": sum(r.turnover for r in reports) / len(reports),
                "net_return_after_fees_taxes": sum(r.net_return for r in reports) / len(reports),
                "config_hash": reports[0].config_hash,
                "dataset_hash": reports[0].dataset_hash,
                "code_hash": reports[0].code_hash,
                "long_exposure": sum(r.long_exposure for r in reports) / len(reports),
                "short_exposure": sum(r.short_exposure for r in reports) / len(reports),
            }
        rows.append(avg)
    rows.sort(key=lambda x: x["sharpe"], reverse=True)
    out = {
        "market": mk,
        "trading_type": normalized_trading_type,
        "leaderboard": rows,
        "warning": "CẢNH BÁO (Warning): Quá khứ không đảm bảo tương lai (Past performance is not indicative of future results); có rủi ro overfit; chi phí thực tế có thể khác mô phỏng.",
    }
    _write_compare_cache(compare_cache_key, out)
    return out


class CreateDraftIn(RunSignalIn):
    pass


@router.post("/create_order_draft")
def create_order_draft(payload: CreateDraftIn) -> dict[str, Any]:
    settings = get_settings()
    mk, position_mode, normalized_trading_type = _resolve_market_mode(
        payload.market, payload.trading_type
    )
    provider = _build_provider(settings, mk, payload.exchange)
    df = provider.get_ohlcv(payload.symbol, payload.timeframe)
    signal = _map_signal_side(
        run_signal(payload.model_id, payload.symbol, payload.timeframe, df), position_mode
    )
    mr = MarketRules.from_yaml(settings.MARKET_RULES_PATH)
    fees = FeesTaxes.from_yaml(settings.FEES_TAXES_PATH)
    draft = generate_order_draft(
        signal=signal,
        market_rules=mr,
        fees_taxes=fees,
        mode=payload.mode,
        allow_short=(mk == "crypto" and normalized_trading_type == "perp_paper"),
        has_open_position=(mk == "crypto" and normalized_trading_type == "perp_paper"),
    )
    return {
        "market": mk,
        "trading_type": normalized_trading_type,
        "signal": signal.model_dump(),
        "draft": None if draft is None else draft.model_dump(),
    }


@router.post("/confirm_execute")
def confirm_execute(
    payload: ConfirmExecutePayload, db: Session = Depends(get_db)
) -> dict[str, Any]:
    ensure_disclaimers(
        acknowledged_educational=payload.acknowledged_educational,
        acknowledged_loss=payload.acknowledged_loss,
        mode=payload.mode,
        acknowledged_live_eligibility=payload.acknowledged_live_eligibility,
        age=payload.age,
    )

    if payload.mode == "live":
        raise HTTPException(
            status_code=422, detail="LiveBrokerStub: mặc định không hoạt động để bảo vệ an toàn"
        )

    portfolio = db.exec(select(Portfolio).where(Portfolio.id == payload.portfolio_id)).first()
    if portfolio is None:
        portfolio = Portfolio(id=payload.portfolio_id, name="Simple Mode Portfolio")
        db.add(portfolio)
        db.flush()

    client_order_id = build_client_order_id(payload.draft.symbol)

    if payload.mode == "paper":
        trade = Trade(
            portfolio_id=payload.portfolio_id,
            trade_date=dt.date.today(),
            symbol=payload.draft.symbol,
            side=payload.draft.side,
            quantity=float(payload.draft.qty),
            price=float(payload.draft.price),
            strategy_tag=f"simple:{payload.draft.mode}",
            notes=client_order_id,
            commission=payload.draft.fee_tax.commission,
            taxes=payload.draft.fee_tax.sell_tax,
            external_id=client_order_id,
        )
        db.add(trade)
        db.flush()
        fill = Fill(
            order_id=int(trade.id or 0),
            execution_id=client_order_id,
            quantity=float(payload.draft.qty),
            price=float(payload.draft.price),
            commission=payload.draft.fee_tax.commission,
            taxes=payload.draft.fee_tax.sell_tax,
        )
        db.add(fill)
        db.commit()
        audit_id = _write_simple_audit(
            {
                "event": "confirm_execute",
                "mode": payload.mode,
                "portfolio_id": payload.portfolio_id,
                "symbol": payload.draft.symbol,
                "side": payload.draft.side,
                "qty": payload.draft.qty,
                "price": payload.draft.price,
                "notional": payload.draft.notional,
                "commission": payload.draft.fee_tax.commission,
                "sell_tax": payload.draft.fee_tax.sell_tax,
                "slippage_est": payload.draft.fee_tax.slippage_est,
                "status": "paper_filled",
                "client_order_id": client_order_id,
            }
        )
        return {
            "status": "paper_filled",
            "trade_id": trade.id,
            "fill_id": fill.id,
            "off_session": payload.draft.off_session,
            "audit_id": audit_id,
        }

    db.commit()
    audit_id = _write_simple_audit(
        {
            "event": "confirm_execute",
            "mode": payload.mode,
            "portfolio_id": payload.portfolio_id,
            "symbol": payload.draft.symbol,
            "side": payload.draft.side,
            "qty": payload.draft.qty,
            "price": payload.draft.price,
            "notional": payload.draft.notional,
            "commission": payload.draft.fee_tax.commission,
            "sell_tax": payload.draft.fee_tax.sell_tax,
            "slippage_est": payload.draft.fee_tax.slippage_est,
            "status": "draft_saved",
            "client_order_id": client_order_id,
        }
    )
    return {
        "status": "draft_saved",
        "draft": payload.draft.model_dump(),
        "off_session": payload.draft.off_session,
        "audit_id": audit_id,
    }


@router.get("/sync_status")
def sync_status(symbol: str = Query(...), timeframe: str = Query(default="1D")) -> dict[str, Any]:
    settings = get_settings()
    provider = get_provider(settings)
    df = provider.get_ohlcv(symbol, timeframe)
    if df.empty:
        return {
            "rows": 0,
            "last_update": None,
            "missing": "Thiếu dữ liệu giá; hãy dùng demo offline (CSV/synthetic) hoặc nạp data_drop.",
        }
    return {
        "rows": len(df),
        "last_update": str(df.iloc[-1].get("date") or df.iloc[-1].get("timestamp")),
        "missing": "",
    }
