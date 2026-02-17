from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from sqlmodel import Session, select

from core.db.models import Fundamental, PriceOHLCV, Ticker
from core.factors import compute_factors
from core.technical import detect_breakout, detect_pullback, detect_trend, detect_volume_spike


@dataclass(frozen=True)
class ScreenDefinition:
    name: str
    description: str
    universe: Dict[str, Any]
    filters: Dict[str, Any]
    factor_weights: Dict[str, float]
    technical_setups: Dict[str, Any]

    @staticmethod
    def from_yaml(path: str | Path) -> "ScreenDefinition":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return ScreenDefinition(
            name=str(data.get("name", "screen")),
            description=str(data.get("description", "")),
            universe=dict(data.get("universe", {}) or {}),
            filters=dict(data.get("filters", {}) or {}),
            factor_weights=dict(data.get("factor_weights", {}) or {}),
            technical_setups=dict(data.get("technical_setups", {}) or {}),
        )


def _apply_universe(tickers_df: pd.DataFrame, universe: Dict[str, Any]) -> pd.DataFrame:
    df = tickers_df.copy()
    ex = universe.get("exchange")
    if ex:
        df = df[df["exchange"].isin(list(ex))]
    return df


def run_screen(session: Session, screen: ScreenDefinition) -> List[Dict[str, Any]]:
    """Run a screen on DB snapshot (MVP, explainable)."""
    tickers = session.exec(select(Ticker)).all()
    fundamentals = session.exec(select(Fundamental)).all()
    prices = session.exec(select(PriceOHLCV).where(PriceOHLCV.timeframe == "1D")).all()

    if not tickers or not fundamentals or not prices:
        return []

    tickers_df = pd.DataFrame([t.model_dump() for t in tickers])
    fund_df = pd.DataFrame([f.model_dump() for f in fundamentals])
    px_df = pd.DataFrame([p.model_dump() for p in prices])

    px_df["date"] = pd.to_datetime(px_df["timestamp"]).dt.date
    px_df = px_df.sort_values(["symbol", "date"])
    px_df["value_vnd"] = px_df["close"].astype(float) * px_df["volume"].astype(float)

    # Universe selection
    tickers_df = _apply_universe(tickers_df, screen.universe)

    # Liquidity proxy: avg value 20D
    liq = (
        px_df.groupby("symbol")["value_vnd"]
        .rolling(20)
        .mean()
        .groupby(level=0)
        .last()
    )

    # catalyst tags
    tags_map = {row["symbol"]: (row.get("tags") or {}) for _, row in tickers_df.iterrows()}

    # compute factors
    out = compute_factors(
        tickers=tickers_df[["symbol", "sector", "is_bank", "is_broker", "shares_outstanding"]].copy(),
        fundamentals=fund_df.copy(),
        prices=px_df[["date", "symbol", "close", "volume", "value_vnd"]].copy(),
        benchmark_symbol="VNINDEX",
    )

    scores = out.scores.copy()
    raw = out.raw_metrics.copy()

    # Attach basic metrics for filtering/scoring
    scores["avg_value_20d"] = liq.reindex(scores.index)
    mc = tickers_df.set_index("symbol")["market_cap"].astype(float)
    scores["market_cap"] = mc.reindex(scores.index)

    # Join raw metrics for filters
    full = scores.join(raw, how="left")

    # Apply filters (basic screener)
    f = screen.filters or {}

    # Sector filter
    if f.get("sector_in"):
        sector_map = tickers_df.set_index("symbol")["sector"].to_dict()
        keep = [sym for sym in full.index if sector_map.get(sym) in set(f["sector_in"])]
        full = full.loc[keep] if keep else full.iloc[0:0]

    if "min_avg_value_20d" in f:
        full = full[full["avg_value_20d"] >= float(f["min_avg_value_20d"])]

    if "market_cap_min" in f:
        full = full[full["market_cap"] >= float(f["market_cap_min"])]

    if "market_cap_max" in f:
        full = full[full["market_cap"] <= float(f["market_cap_max"])]

    if "max_pe" in f:
        full = full[full["PE"] <= float(f["max_pe"])]

    if "max_pb" in f:
        full = full[full["PB"] <= float(f["max_pb"])]

    if "min_roe" in f:
        full = full[full["ROE"] >= float(f["min_roe"])]

    if "max_net_debt_to_ebitda" in f:
        # apply only non-bank (banks do not use net debt/EBITDA)
        is_bank = tickers_df.set_index("symbol")["is_bank"].astype(bool)
        cap = float(f["max_net_debt_to_ebitda"])
        mask = (~is_bank.reindex(full.index).fillna(False)) & (full["NET_DEBT_TO_EBITDA"] <= cap)
        # keep banks regardless of this filter (MVP)
        full = full[mask | is_bank.reindex(full.index).fillna(False)]

    # Catalyst tags (ANY)
    if "catalyst_tags_any" in f:
        need = set([str(x) for x in (f.get("catalyst_tags_any") or [])])
        keep = []
        for sym in full.index:
            t = tags_map.get(sym, {})
            cats = set([str(x) for x in (t.get("catalysts") or [])])
            if not need or (cats & need):
                keep.append(sym)
        full = full.loc[keep] if keep else full.iloc[0:0]

    # Total factor scoring
    weights = screen.factor_weights or {}
    full["total_factor"] = 0.0
    for k, w in weights.items():
        if k in full.columns:
            full["total_factor"] += float(w) * full[k].fillna(0.0)

    # Technical setups
    results: List[Dict[str, Any]] = []
    for sym in full.index:
        g = px_df[px_df["symbol"] == sym].copy()
        g = g.sort_values("timestamp").tail(260)
        g = g.set_index(pd.to_datetime(g["timestamp"]))[["open", "high", "low", "close", "volume"]]
        if g.empty:
            continue

        setups: Dict[str, bool] = {
            "breakout": False,
            "trend": False,
            "pullback": False,
            "volume_spike": False,
        }

        ts = screen.technical_setups or {}
        if "breakout" in ts:
            lookback = int((ts["breakout"] or {}).get("lookback", 20))
            vm = float((ts["breakout"] or {}).get("volume_multiple", 1.5))
            setups["breakout"] = detect_breakout(g, lookback=lookback, volume_multiple=vm)

        setups["trend"] = detect_trend(g)
        setups["pullback"] = detect_pullback(g)
        setups["volume_spike"] = detect_volume_spike(g)

        total = float(full.loc[sym, "total_factor"])
        # small bonus for setups (MVP)
        total += 0.2 if setups["breakout"] else 0.0
        total += 0.1 if setups["trend"] else 0.0
        total += 0.05 if setups["pullback"] else 0.0
        total += 0.05 if setups["volume_spike"] else 0.0

        breakdown = {k: float(full.loc[sym, k]) for k in weights.keys() if k in full.columns}
        explain = {
            "filters_snapshot": {
                "avg_value_20d": _safe_float(full.loc[sym, "avg_value_20d"]),
                "market_cap": _safe_float(full.loc[sym, "market_cap"]),
                "PE": _safe_float(full.loc[sym, "PE"]),
                "PB": _safe_float(full.loc[sym, "PB"]),
                "ROE": _safe_float(full.loc[sym, "ROE"]),
                "NET_DEBT_TO_EBITDA": _safe_float(full.loc[sym, "NET_DEBT_TO_EBITDA"]),
            },
            "setups": setups,
            "tags": tags_map.get(sym, {}),
        }

        results.append(
            {
                "symbol": sym,
                "total_score": total,
                "factor_breakdown": breakdown,
                "setups": setups,
                "explain": explain,
            }
        )

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results


def _safe_float(x) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None
