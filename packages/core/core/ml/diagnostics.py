from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rank_ic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("as_of_date"):
        if g["score_final"].nunique() < 2 or g["y_excess"].nunique() < 2:
            ic = 0.0
        else:
            ic = float(g[["score_final", "y_excess"]].corr(method="spearman").iloc[0, 1])
        rows.append({"as_of_date": d, "rank_ic": ic})
    return pd.DataFrame(rows)


def ic_decay(df: pd.DataFrame, ks: list[int] | None = None) -> dict[str, float]:
    ks = ks or [1, 5, 21, 63]
    out: dict[str, float] = {}
    for k in ks:
        shifted = df.copy().sort_values(["symbol", "as_of_date"])
        shifted["fwd"] = shifted.groupby("symbol")["y_excess"].shift(-k)
        ic_df = rank_ic(shifted.rename(columns={"fwd": "y_excess"}))
        out[f"ic_decay_{k}"] = float(ic_df["rank_ic"].mean()) if not ic_df.empty else 0.0
    return out


def decile_spread(df: pd.DataFrame) -> float:
    spreads = []
    for _, g in df.groupby("as_of_date"):
        h = g.copy()
        h["decile"] = pd.qcut(h["score_final"].rank(method="first"), 10, labels=False, duplicates="drop")
        top = h[h["decile"] == h["decile"].max()]["net_ret"].mean()
        bot = h[h["decile"] == h["decile"].min()]["net_ret"].mean()
        spreads.append(float(top - bot))
    return float(np.nanmean(spreads)) if spreads else 0.0


def turnover_cost_attribution(df: pd.DataFrame) -> dict[str, float]:
    return {
        "turnover": float(df.get("turnover", pd.Series([0.0])).sum()),
        "commission": float(df.get("commission", pd.Series([0.0])).sum()),
        "sell_tax": float(df.get("sell_tax", pd.Series([0.0])).sum()),
        "slippage_cost": float(df.get("slippage_cost", pd.Series([0.0])).sum()),
    }


def capacity_proxy(df: pd.DataFrame) -> dict[str, float]:
    ratio = (df.get("order_notional", 0.0) / df.get("adv20_value", 1.0)).replace([np.inf, -np.inf], np.nan)
    bind = (df.get("liq_bound", False)).astype(float)
    return {
        "avg_order_notional_over_adtv": float(np.nanmean(ratio)),
        "liq_constraint_bind_pct": float(np.nanmean(bind)),
    }


def regime_breakdown(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ["trend_up", "sideways", "risk_off"]:
        m = df[df.get("regime", "sideways") == k]["net_ret"]
        out[f"{k}_mean_ret"] = float(m.mean()) if len(m) else 0.0
    return out


def block_bootstrap_ci(returns: pd.Series, block: int = 20, n_resamples: int = 1000) -> dict[str, float]:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return {"sharpe_lo": 0.0, "sharpe_hi": 0.0, "cagr_lo": 0.0, "cagr_hi": 0.0}
    rng = np.random.default_rng(42)
    sharpes, cagrs = [], []
    n_blocks = math.ceil(len(r) / block)
    for _ in range(n_resamples):
        idx = []
        for _ in range(n_blocks):
            s = rng.integers(0, max(1, len(r) - block + 1))
            idx.extend(range(s, min(len(r), s + block)))
        b = r[np.array(idx[: len(r)])]
        sharpe = float((b.mean() / (b.std() + 1e-12)) * np.sqrt(252))
        eq = float(np.prod(1.0 + b))
        years = max(len(b) / 252.0, 1 / 252.0)
        cagr = -1.0 if eq <= 0 else float(eq ** (1 / years) - 1.0)
        sharpes.append(sharpe)
        cagrs.append(cagr)
    return {
        "sharpe_lo": float(np.quantile(sharpes, 0.025)),
        "sharpe_hi": float(np.quantile(sharpes, 0.975)),
        "cagr_lo": float(np.quantile(cagrs, 0.025)),
        "cagr_hi": float(np.quantile(cagrs, 0.975)),
    }
