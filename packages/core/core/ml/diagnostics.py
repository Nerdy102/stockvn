from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rank_ic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("as_of_date"):
        x = g["score_final"]
        y = g["y_excess"]
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            ic = 0.0
        else:
            ic = float(g[["score_final", "y_excess"]].corr(method="spearman").iloc[0, 1])
        rows.append({"as_of_date": d, "rank_ic": ic})
    return pd.DataFrame(rows)


def ic_decay(df: pd.DataFrame, horizons: list[int] | None = None) -> dict[str, float]:
    horizons = horizons or [1, 5, 21, 63]
    out: dict[str, float] = {}
    base = df.sort_values(["symbol", "as_of_date"]).copy()
    for h in horizons:
        tmp = base.copy()
        tmp[f"y_excess_h{h}"] = tmp.groupby("symbol")["y_excess"].shift(-h)
        ic = rank_ic(tmp.rename(columns={f"y_excess_h{h}": "y_excess"}))
        out[f"ic_decay_{h}"] = float(ic["rank_ic"].mean()) if not ic.empty else 0.0
    return out


def deciles(df: pd.DataFrame) -> dict[str, float]:
    spreads = []
    for _, g in df.groupby("as_of_date"):
        t = g.dropna(subset=["score_final", "net_ret"]).copy()
        if len(t) < 10:
            continue
        t["dec"] = pd.qcut(t["score_final"].rank(method="first"), 10, labels=False)
        top = float(t[t["dec"] == t["dec"].max()]["net_ret"].mean())
        bot = float(t[t["dec"] == t["dec"].min()]["net_ret"].mean())
        spreads.append(top - bot)
    return {
        "decile_spread_net": float(np.mean(spreads)) if spreads else 0.0,
        "decile_obs": float(len(spreads)),
    }


def turnover_cost_attribution(df: pd.DataFrame) -> dict[str, float]:
    return {
        "turnover": float(df.get("turnover", pd.Series([0.0])).sum()),
        "commission": float(df.get("commission", pd.Series([0.0])).sum()),
        "sell_tax": float(df.get("sell_tax", pd.Series([0.0])).sum()),
        "slippage_cost": float(df.get("slippage_cost", pd.Series([0.0])).sum()),
    }


def capacity(df: pd.DataFrame) -> dict[str, float]:
    ratio = (df.get("order_notional", 0.0) / df.get("adv20_value", 1.0)).replace([np.inf, -np.inf], np.nan)
    binds = df.get("liq_bound", False).astype(float)
    return {
        "capacity_avg_order_notional_over_adtv": float(np.nanmean(ratio)) if len(df) else 0.0,
        "capacity_liq_bind_pct": float(np.nanmean(binds)) if len(df) else 0.0,
    }


def regime_breakdown(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in ["trend_up", "sideways", "risk_off"]:
        v = df[df.get("regime", "sideways") == r]["net_ret"]
        out[f"regime_{r}_mean_net_ret"] = float(v.mean()) if len(v) else 0.0
    return out


def block_bootstrap_ci(returns: pd.Series, block: int = 20, n_resamples: int = 1000) -> dict[str, float]:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return {"sharpe_lo": 0.0, "sharpe_hi": 0.0, "cagr_lo": 0.0, "cagr_hi": 0.0}

    rng = np.random.default_rng(42)
    ns = math.ceil(len(r) / block)
    sharpe_s, cagr_s = [], []
    for _ in range(n_resamples):
        idx: list[int] = []
        for _ in range(ns):
            s = int(rng.integers(0, max(1, len(r) - block + 1)))
            idx.extend(list(range(s, min(len(r), s + block))))
        b = r[np.array(idx[: len(r)])]
        sharpe = float((b.mean() / (b.std() + 1e-12)) * np.sqrt(252))
        eq = float(np.prod(1.0 + b))
        years = max(len(b) / 252.0, 1 / 252.0)
        cagr = -1.0 if eq <= 0 else float(eq ** (1.0 / years) - 1.0)
        sharpe_s.append(sharpe)
        cagr_s.append(cagr)
    return {
        "sharpe_lo": float(np.quantile(sharpe_s, 0.025)),
        "sharpe_hi": float(np.quantile(sharpe_s, 0.975)),
        "cagr_lo": float(np.quantile(cagr_s, 0.025)),
        "cagr_hi": float(np.quantile(cagr_s, 0.975)),
    }


def run_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    ic = rank_ic(df)
    metrics: dict[str, float] = {
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else 0.0,
    }
    metrics.update(ic_decay(df))
    metrics.update(deciles(df))
    metrics.update(turnover_cost_attribution(df))
    metrics.update(capacity(df))
    metrics.update(regime_breakdown(df))
    metrics.update(block_bootstrap_ci(df.get("net_ret", pd.Series(dtype=float))))
    return metrics
