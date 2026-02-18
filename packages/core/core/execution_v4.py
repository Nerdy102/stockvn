from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


LOT_SIZE = 100
SESSION_OPEN_AUCTION = "open_auction"
SESSION_CONTINUOUS_AM = "continuous_am"
SESSION_LUNCH = "lunch_break"
SESSION_CONTINUOUS_PM = "continuous_pm"
SESSION_CLOSE_AUCTION = "close_auction"
SESSION_OFFHOURS = "offhours"

SESSION_WINDOWS = {
    SESSION_OPEN_AUCTION: ((9, 0), (9, 15)),
    SESSION_CONTINUOUS_AM: ((9, 15), (11, 30)),
    SESSION_LUNCH: ((11, 30), (13, 0)),
    SESSION_CONTINUOUS_PM: ((13, 0), (14, 30)),
    SESSION_CLOSE_AUCTION: ((14, 30), (14, 45)),
}


@dataclass(frozen=True)
class ExecutionV4Assumptions:
    base_slippage_bps: float = 10.0
    participation_limit: float = 0.05
    carryover_limit_ratio: float = 0.30
    fill_ratio_open_auction: float = 0.35
    fill_ratio_continuous: float = 0.90
    fill_ratio_close_auction: float = 0.50
    fill_ratio_offhours_fallback: float = 0.10
    slippage_addon_open_auction_bps: float = 8.0
    slippage_addon_continuous_bps: float = 2.0
    slippage_addon_close_auction_bps: float = 6.0
    slippage_addon_offhours_bps: float = 12.0


def classify_hose_session(ts: pd.Timestamp | str) -> str:
    t = pd.Timestamp(ts)
    hhmm = (int(t.hour), int(t.minute))

    def _in_window(start: tuple[int, int], end: tuple[int, int]) -> bool:
        return start <= hhmm < end

    for session, (start, end) in SESSION_WINDOWS.items():
        if _in_window(start, end):
            return session
    return SESSION_OFFHOURS


def compute_session_vwap(bars: pd.DataFrame) -> pd.DataFrame:
    work = bars.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work["session"] = work["timestamp"].map(classify_hose_session)
    work["price_ref"] = pd.to_numeric(work.get("close"), errors="coerce")
    if "open" in work.columns:
        work["price_ref"] = work["price_ref"].fillna(pd.to_numeric(work["open"], errors="coerce"))
    work["price_ref"] = work["price_ref"].fillna(0.0)
    work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0)

    grouped = (
        work.groupby("session", as_index=False)
        .agg(
            session_vwap_num=("price_ref", lambda x: 0.0),
            session_volume=("volume", "sum"),
            n_bars=("session", "size"),
        )
        .copy()
    )
    # deterministic VWAP numerator computed with aligned weights
    vwap_num = work.assign(vwap_num=work["price_ref"] * work["volume"]).groupby("session")["vwap_num"].sum()
    grouped["session_vwap"] = grouped["session"].map(vwap_num).astype(float) / grouped["session_volume"].clip(lower=1.0)
    grouped = grouped.drop(columns=["session_vwap_num"])
    return grouped[["session", "session_vwap", "session_volume", "n_bars"]].sort_values("session").reset_index(drop=True)


def _session_fill_ratio(session: str, assumptions: ExecutionV4Assumptions) -> float:
    if session == SESSION_OPEN_AUCTION:
        return assumptions.fill_ratio_open_auction
    if session in (SESSION_CONTINUOUS_AM, SESSION_CONTINUOUS_PM):
        return assumptions.fill_ratio_continuous
    if session == SESSION_CLOSE_AUCTION:
        return assumptions.fill_ratio_close_auction
    return assumptions.fill_ratio_offhours_fallback


def _session_slippage_addon_bps(session: str, assumptions: ExecutionV4Assumptions) -> float:
    if session == SESSION_OPEN_AUCTION:
        return assumptions.slippage_addon_open_auction_bps
    if session in (SESSION_CONTINUOUS_AM, SESSION_CONTINUOUS_PM):
        return assumptions.slippage_addon_continuous_bps
    if session == SESSION_CLOSE_AUCTION:
        return assumptions.slippage_addon_close_auction_bps
    return assumptions.slippage_addon_offhours_bps


def _round_lot(qty: float, lot_size: int = LOT_SIZE) -> int:
    return int(max(0, np.floor(float(qty) / lot_size) * lot_size))


def simulate_execution_v4(
    bars: pd.DataFrame,
    *,
    symbol: str,
    side: str,
    order_qty: int,
    assumptions: ExecutionV4Assumptions = ExecutionV4Assumptions(),
) -> dict[str, Any]:
    if order_qty <= 0:
        raise ValueError("order_qty must be positive")
    s = str(side).upper()
    if s not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")

    work = bars.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        return {"fills": pd.DataFrame(), "tca_log": pd.DataFrame(), "summary": {}}

    work["session"] = work["timestamp"].map(classify_hose_session)
    work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0)
    work["price_ref"] = pd.to_numeric(work.get("close"), errors="coerce")
    if "open" in work.columns:
        work["price_ref"] = work["price_ref"].fillna(pd.to_numeric(work["open"], errors="coerce"))
    work["price_ref"] = work["price_ref"].ffill().fillna(0.0)
    work["spread_bps"] = pd.to_numeric(work.get("spread_bps", 0.0), errors="coerce").fillna(0.0)

    remaining = _round_lot(order_qty)
    initial_qty = remaining
    fills: list[dict[str, Any]] = []
    tca_logs: list[dict[str, Any]] = []

    for row in work.itertuples(index=False):
        if remaining <= 0:
            break
        session = str(row.session)
        price_ref = float(row.price_ref)
        volume = float(row.volume)
        spread_bps = max(0.0, float(row.spread_bps))

        cap_qty_raw = max(0.0, volume * assumptions.participation_limit)
        cap_qty = _round_lot(cap_qty_raw)
        fill_ratio = _session_fill_ratio(session, assumptions)
        fillable_qty = _round_lot(min(float(remaining), cap_qty_raw) * fill_ratio)

        addon_bps = _session_slippage_addon_bps(session, assumptions)
        total_slippage_bps = float(max(0.0, assumptions.base_slippage_bps + addon_bps + spread_bps))
        slippage_mult = total_slippage_bps / 10000.0
        exec_price = price_ref * (1.0 + slippage_mult if s == "BUY" else 1.0 - slippage_mult)

        filled_qty = int(min(remaining, fillable_qty))
        notional = float(filled_qty) * float(exec_price)

        tca_logs.append(
            {
                "symbol": symbol,
                "timestamp": pd.Timestamp(row.timestamp),
                "session": session,
                "side": s,
                "order_qty": int(initial_qty),
                "remaining_before": int(remaining),
                "bar_volume": float(volume),
                "participation_limit": float(assumptions.participation_limit),
                "cap_qty": int(cap_qty),
                "fill_ratio_rule": float(fill_ratio),
                "filled_qty": int(filled_qty),
                "remaining_after": int(max(0, remaining - filled_qty)),
                "price_ref": float(price_ref),
                "base_slippage_bps": float(assumptions.base_slippage_bps),
                "session_slippage_addon_bps": float(addon_bps),
                "spread_addon_bps": float(spread_bps),
                "total_slippage_bps": float(total_slippage_bps),
                "exec_price": float(exec_price),
                "exec_notional": float(notional),
            }
        )

        if filled_qty > 0:
            fills.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp(row.timestamp),
                    "session": session,
                    "side": s,
                    "filled_qty": int(filled_qty),
                    "exec_price": float(exec_price),
                    "exec_notional": float(notional),
                }
            )
            remaining -= filled_qty

    carry_limit = _round_lot(initial_qty * assumptions.carryover_limit_ratio)
    carryover_qty = min(remaining, carry_limit)
    cancelled_qty = max(0, remaining - carryover_qty)
    executed_qty = int(initial_qty - remaining)

    summary = {
        "symbol": symbol,
        "side": s,
        "order_qty": int(initial_qty),
        "executed_qty": int(executed_qty),
        "carryover_qty": int(carryover_qty),
        "cancelled_qty": int(cancelled_qty),
        "carryover_limit_qty": int(carry_limit),
        "fills_count": int(len(fills)),
        "exec_notional_total": float(sum(f["exec_notional"] for f in fills)),
        "recon_qty_ok": bool(initial_qty == executed_qty + carryover_qty + cancelled_qty),
    }

    return {
        "fills": pd.DataFrame(fills),
        "tca_log": pd.DataFrame(tca_logs),
        "summary": summary,
        "session_vwap": compute_session_vwap(work),
    }
