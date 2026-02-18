from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.alpha_v3.portfolio import construct_portfolio_v3_with_report
from core.corporate_actions import adjust_prices


def _load_cfg() -> dict:
    return json.loads(Path("tests/golden/metamorphic_invariants_quant_v4_08.json").read_text(encoding="utf-8"))


def _portfolio_inputs(n_assets: int = 8, t: int = 220) -> dict[str, object]:
    rng = np.random.default_rng(202508)
    symbols = [f"S{i:02d}" for i in range(n_assets)]
    returns = rng.normal(0.0002, 0.01, size=(t, n_assets))
    scores = np.linspace(1.0, 0.0, n_assets)
    current_w = np.full(n_assets, 1.0 / n_assets)
    next_open = np.full(n_assets, 20_000.0)
    adtv = np.linspace(2e9, 8e9, n_assets)
    atr14 = np.full(n_assets, 180.0)
    close = np.full(n_assets, 20_000.0)
    spread = np.full(n_assets, 0.0006)
    sectors = np.array(["A", "A", "B", "B", "C", "C", "D", "D"], dtype=object)
    return {
        "symbols": symbols,
        "returns": returns,
        "scores": scores,
        "current_w": current_w,
        "next_open": next_open,
        "adtv": adtv,
        "atr14": atr14,
        "close": close,
        "spread": spread,
        "sectors": sectors,
    }


def _weights_by_symbol(weights: np.ndarray, report: dict[str, object]) -> dict[str, float]:
    syms = list(report["universe"]["symbols"])
    return {str(s): float(w) for s, w in zip(syms, np.asarray(weights, dtype=float))}


def test_metamorphic_symbol_order_permutation_invariance() -> None:
    cfg = _load_cfg()
    atol = float(cfg["tolerances"]["weight_atol"])
    p = _portfolio_inputs()

    w0, _c0, _i0, r0 = construct_portfolio_v3_with_report(
        p["symbols"],
        p["returns"],
        p["current_w"],
        1_000_000_000.0,
        p["next_open"],
        p["adtv"],
        p["atr14"],
        p["close"],
        p["spread"],
        p["sectors"],
        scores=p["scores"],
        top_k=6,
    )
    base_map = _weights_by_symbol(w0, r0)

    perm = np.array([3, 1, 7, 0, 6, 2, 5, 4])
    symbols_p = [p["symbols"][int(i)] for i in perm]
    w1, _c1, _i1, r1 = construct_portfolio_v3_with_report(
        symbols_p,
        p["returns"][:, perm],
        p["current_w"][perm],
        1_000_000_000.0,
        p["next_open"][perm],
        p["adtv"][perm],
        p["atr14"][perm],
        p["close"][perm],
        p["spread"][perm],
        p["sectors"][perm],
        scores=p["scores"][perm],
        top_k=6,
    )
    perm_map = _weights_by_symbol(w1, r1)

    assert set(base_map) == set(perm_map)
    for sym in sorted(base_map):
        assert abs(base_map[sym] - perm_map[sym]) <= atol


def test_metamorphic_score_shift_invariance() -> None:
    cfg = _load_cfg()
    atol = float(cfg["tolerances"]["weight_atol"])
    p = _portfolio_inputs()

    w0, _c0, _i0, r0 = construct_portfolio_v3_with_report(
        p["symbols"],
        p["returns"],
        p["current_w"],
        1_000_000_000.0,
        p["next_open"],
        p["adtv"],
        p["atr14"],
        p["close"],
        p["spread"],
        p["sectors"],
        scores=p["scores"],
        top_k=6,
    )
    w1, _c1, _i1, r1 = construct_portfolio_v3_with_report(
        p["symbols"],
        p["returns"],
        p["current_w"],
        1_000_000_000.0,
        p["next_open"],
        p["adtv"],
        p["atr14"],
        p["close"],
        p["spread"],
        p["sectors"],
        scores=p["scores"] + 1000.0,
        top_k=6,
    )

    assert r0["universe"]["symbols"] == r1["universe"]["symbols"]
    assert np.allclose(np.asarray(w0, dtype=float), np.asarray(w1, dtype=float), atol=atol)


def test_metamorphic_price_scaling_invariance_returns() -> None:
    cfg = _load_cfg()
    atol = float(cfg["tolerances"]["return_atol"])

    px = pd.Series([100.0, 101.5, 99.0, 100.0, 103.0, 102.5], dtype=float)
    ret = px.pct_change().fillna(0.0)
    ret_scaled = (px * 100.0).pct_change().fillna(0.0)
    assert np.allclose(ret.to_numpy(), ret_scaled.to_numpy(), atol=atol)


def test_metamorphic_split_ca_adjusted_series_invariance() -> None:
    cfg = _load_cfg()
    atol = float(cfg["tolerances"]["return_atol"])

    dates = [dt.date(2025, 1, d) for d in range(1, 6)]
    baseline_close = [100.0, 102.0, 104.0, 106.0, 108.0]
    baseline = pd.DataFrame(
        {
            "date": dates,
            "open": baseline_close,
            "high": baseline_close,
            "low": baseline_close,
            "close": baseline_close,
            "volume": [1000.0] * 5,
        }
    )

    split_raw_close = [100.0, 102.0, 52.0, 53.0, 54.0]
    split_raw = baseline.copy()
    split_raw["open"] = split_raw_close
    split_raw["high"] = split_raw_close
    split_raw["low"] = split_raw_close
    split_raw["close"] = split_raw_close
    split_raw["volume"] = [1000.0, 1000.0, 2000.0, 2000.0, 2000.0]

    ca = [
        {
            "symbol": "AAA",
            "action_type": "SPLIT",
            "ex_date": dt.date(2025, 1, 3),
            "params_json": {"split_factor": 2.0},
        }
    ]
    adjusted = adjust_prices(
        "AAA",
        split_raw,
        start=dt.date(2025, 1, 1),
        end=dt.date(2025, 1, 5),
        method="ca",
        corporate_actions=ca,
    )

    base_ret = baseline["close"].pct_change().fillna(0.0)
    adj_ret = adjusted["close"].pct_change().fillna(0.0)
    assert np.allclose(base_ret.to_numpy(), adj_ret.to_numpy(), atol=atol)
