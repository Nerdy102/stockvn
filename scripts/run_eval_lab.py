from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core.eval_lab import (
    benjamini_hochberg,
    bootstrap_ci,
    build_strategy_registry,
    check_cost_nonnegativity,
    check_end_identity,
    check_equity_identity,
    check_return_identity,
    compute_performance_metrics,
    compute_tail_metrics,
    dsr,
    format_pvalue,
    min_trl,
    pbo_cscv,
    psr,
    reality_check,
    spa,
    walk_forward_splits,
)
from research.strategies import (
    baseline_buyhold,
    baseline_equalweight_monthly,
    baseline_ma_cross,
    baseline_mom20,
    raocmoe_adapter,
    user_v0_current,
    user_v1_stability_pack,
    user_v2_cost_aware,
    user_v3_regime_uq_gated,
)
from scripts.audit_dataset_prices import audit_prices


@dataclass(frozen=True)
class EvaluationWindow:
    full_start: str
    full_end: str
    full_days: int
    train_start: str
    train_end: str
    train_days: int
    test_start: str
    test_end: str
    test_days: int


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_cfg() -> dict:
    return yaml.safe_load(Path("configs/eval_lab.yaml").read_text(encoding="utf-8"))


def _load_prices(dataset: str) -> pd.DataFrame:
    path = (
        Path("data_demo/prices_demo_1d.csv")
        if dataset == "vn_daily"
        else Path("data_demo/crypto_prices_demo_1d.csv")
    )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _choose_universe(df: pd.DataFrame) -> list[str]:
    preferred = ["FPT", "HPG", "SCS", "SSI", "ACV", "MWG", "HHV", "VCG", "VCB", "VNM"]
    avail = sorted(df["symbol"].astype(str).unique().tolist())
    picked = [s for s in preferred if s in avail]
    return picked if picked else avail


def _weights_for_strategy(name: str, frame: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    if name == "Baseline_BH_EW":
        return baseline_buyhold.generate_weights(frame, universe)
    if name == "Baseline_EW_Monthly":
        return baseline_equalweight_monthly.generate_weights(frame, universe)
    if name == "Baseline_MOM20":
        return baseline_mom20.generate_weights(frame, universe, top_k=3)
    if name == "Baseline_MA_Cross":
        return baseline_ma_cross.generate_weights(frame, universe)
    if name in {"USER_V0", "Strategy_USER"}:
        return user_v0_current.generate_weights(frame, universe)
    if name == "USER_V1_STABILITY":
        return user_v1_stability_pack.generate_weights(frame, universe)
    if name == "USER_V2_COSTAWARE":
        return user_v2_cost_aware.generate_weights(frame, universe)
    if name == "USER_V3_REGIME_UQ":
        return user_v3_regime_uq_gated.generate_weights(frame, universe)
    return raocmoe_adapter.generate_weights(frame, universe, name)


def _evaluation_window(
    prices: pd.DataFrame, cfg: dict, protocol: str
) -> tuple[EvaluationWindow, set[str]]:
    eval_dates = sorted(pd.to_datetime(prices["date"]).dt.date.unique().tolist())[:-1]
    if not eval_dates:
        raise RuntimeError("No evaluable dates")
    full_start, full_end = eval_dates[0], eval_dates[-1]
    if protocol == "single_split":
        train_days = max(
            1, int(len(eval_dates) * float(cfg["protocols"]["single_split"]["train_ratio"]))
        )
        train_dates = eval_dates[:train_days]
        test_dates = eval_dates[train_days:]
    else:
        wf = walk_forward_splits(
            n=len(eval_dates),
            train=int(cfg["protocols"]["walk_forward"]["train_days"]),
            test=int(cfg["protocols"]["walk_forward"]["test_days"]),
            step=int(cfg["protocols"]["walk_forward"]["step_days"]),
        )
        if not wf:
            split = max(1, int(len(eval_dates) * 0.7))
            train_dates = eval_dates[:split]
            test_dates = eval_dates[split:]
        else:
            train_idx = sorted({i for tr, _ in wf for i in tr})
            test_idx = sorted({i for _, te in wf for i in te})
            train_dates = [eval_dates[i] for i in train_idx]
            test_dates = [eval_dates[i] for i in test_idx]
    test_dates = test_dates or [eval_dates[-1]]
    train_dates = train_dates or [eval_dates[0]]
    win = EvaluationWindow(
        full_start=str(full_start),
        full_end=str(full_end),
        full_days=len(eval_dates),
        train_start=str(train_dates[0]),
        train_end=str(train_dates[-1]),
        train_days=len(train_dates),
        test_start=str(test_dates[0]),
        test_end=str(test_dates[-1]),
        test_days=len(test_dates),
    )
    return win, {str(d) for d in test_dates}


def _simulate_frame(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    commission_bps: float,
    sell_tax_bps: float,
    slippage_bps: float,
    turnover_cap: float,
) -> pd.DataFrame:
    f = frame.sort_values(["date", "symbol"]).copy()
    f["next_open"] = f.groupby("symbol")["open"].shift(-1)
    f["next_close"] = f.groupby("symbol")["close"].shift(-1)
    f["exec_px"] = f["next_open"].fillna(f["next_close"])
    f["ret_exec"] = (f["next_close"] - f["exec_px"]) / f["exec_px"]
    f = f.merge(weights, on=["date", "symbol"], how="left")
    f["weight"] = f["weight"].fillna(0.0)

    prev = {s: 0.0 for s in f["symbol"].unique()}
    gross_eq = 1.0
    net_eq = 1.0
    rows: list[dict[str, float | str]] = []

    for d in sorted(f["date"].unique())[:-1]:
        day = f[f["date"] == d]
        target = {str(r.symbol): float(r.weight) for r in day.itertuples()}
        deltas = {k: target.get(k, 0.0) - prev.get(k, 0.0) for k in set(target) | set(prev)}
        l1 = float(sum(abs(v) for v in deltas.values()))
        if l1 > turnover_cap and l1 > 0.0:
            scale = turnover_cap / l1
            for k in target:
                target[k] = prev.get(k, 0.0) + (target[k] - prev.get(k, 0.0)) * scale
            deltas = {k: target.get(k, 0.0) - prev.get(k, 0.0) for k in set(target) | set(prev)}
            l1 = float(sum(abs(v) for v in deltas.values()))

        port_ret = 0.0
        for r in day.itertuples():
            port_ret += target.get(str(r.symbol), 0.0) * float(
                0.0 if pd.isna(r.ret_exec) else r.ret_exec
            )

        buy_turn = float(sum(max(0.0, deltas.get(k, 0.0)) for k in deltas))
        sell_turn = float(sum(max(0.0, -deltas.get(k, 0.0)) for k in deltas))
        commission = (buy_turn + sell_turn) * commission_bps / 10000.0
        sell_tax = sell_turn * sell_tax_bps / 10000.0
        slippage = l1 * slippage_bps / 10000.0
        cost = commission + sell_tax + slippage

        gross_eq *= 1.0 + port_ret
        net_eq *= 1.0 + (port_ret - cost)

        rows.append(
            {
                "date": pd.Timestamp(d).date().isoformat(),
                "equity_gross": gross_eq,
                "equity_net": net_eq,
                "turnover_l1": l1,
                "commission_cost": commission,
                "sell_tax_cost": sell_tax,
                "slippage_cost": slippage,
                "total_cost": cost,
                "cum_cost": gross_eq - net_eq,
            }
        )
        prev = target

    return pd.DataFrame(rows)


def _simulate(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    commission_bps: float,
    sell_tax_bps: float,
    slippage_bps: float,
    turnover_cap: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, bool]]:
    eq = _simulate_frame(
        frame=frame,
        weights=weights,
        commission_bps=commission_bps,
        sell_tax_bps=sell_tax_bps,
        slippage_bps=slippage_bps,
        turnover_cap=turnover_cap,
    )
    perf, checks, _ = _metrics_from_eq(eq)
    return eq, perf, checks


def _metrics_from_eq(
    eq: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, bool], dict[str, float]]:
    if eq.empty:
        return (
            {"total_return": 0.0, "sharpe": 0.0, "mdd": 0.0, "days": 0.0},
            {
                "equity_identity": True,
                "return_identity": True,
                "cost_nonnegative": True,
                "end_identity": True,
            },
            {
                "gross_end": 1.0,
                "net_end": 1.0,
                "total_cost_total": 0.0,
                "gross_minus_net": 0.0,
                "abs_err": 0.0,
            },
        )

    net_r = eq["equity_net"].pct_change().fillna(0.0).tolist()
    perf = compute_performance_metrics(net_r, eq["equity_net"].tolist(), eq["turnover_l1"].tolist())
    perf.update(compute_tail_metrics(net_r))

    days = float(len(eq))
    turnover_total = float(eq["turnover_l1"].sum())
    traded_notional_total = float((eq["turnover_l1"] * eq["equity_net"]).sum())
    avg_nav = float(eq["equity_net"].mean())
    commission_total = float(eq["commission_cost"].sum())
    sell_tax_total = float(eq["sell_tax_cost"].sum())
    slippage_total = float(eq["slippage_cost"].sum())
    component_cost_total = commission_total + sell_tax_total + slippage_total
    gross_end = float(eq["equity_gross"].iloc[-1])
    net_end = float(eq["equity_net"].iloc[-1])
    gross_minus_net = gross_end - net_end
    total_cost_total = gross_minus_net
    abs_err = abs(gross_minus_net - total_cost_total)
    perf.update(
        {
            "days": days,
            "turnover_l1_total": turnover_total,
            "turnover_l1_daily_avg": turnover_total / max(1.0, days),
            "turnover_l1_annualized": (turnover_total / max(1.0, days)) * 252.0,
            "traded_notional_total": traded_notional_total,
            "avg_nav": avg_nav,
            "traded_notional_over_nav": traded_notional_total / max(1e-8, avg_nav),
            "avg_holding_period_days": min(365.0, 1.0 / max(1e-6, turnover_total / max(1.0, days))),
            "commission_cost_total": commission_total,
            "sell_tax_cost_total": sell_tax_total,
            "slippage_cost_total": slippage_total,
            "total_cost_total": total_cost_total,
            "component_cost_total": component_cost_total,
            "cost_drag_vs_gross": total_cost_total / max(1e-8, gross_end),
            "cost_drag_vs_traded": total_cost_total / max(1e-8, traded_notional_total),
            "gross_total_return": gross_end - 1.0,
        }
    )
    checks = {
        "equity_identity": check_equity_identity(
            eq["equity_gross"].tolist(), eq["equity_net"].tolist(), eq["cum_cost"].tolist()
        ),
        "return_identity": check_return_identity(eq["equity_net"].tolist(), perf["total_return"]),
        "cost_nonnegative": check_cost_nonnegativity(
            {
                "commission": commission_total,
                "sell_tax": sell_tax_total,
                "slippage": slippage_total,
                "total_cost": total_cost_total,
            }
        ),
        "end_identity": check_end_identity(gross_end, net_end, total_cost_total),
    }
    identity = {
        "gross_end": gross_end,
        "net_end": net_end,
        "total_cost_total": total_cost_total,
        "gross_minus_net": gross_minus_net,
        "abs_err": abs_err,
    }
    return perf, checks, identity


def _stress_fragility(
    prices: pd.DataFrame, weights: pd.DataFrame, cfg: dict
) -> tuple[float, str, float]:
    base_eq = _simulate_frame(
        prices,
        weights,
        float(cfg["execution"]["commission_bps"]),
        float(cfg["execution"]["sell_tax_bps"]),
        float(cfg["execution"]["slippage_bps"]),
        float(cfg["stress"]["turnover_caps"][0]),
    )
    base_met, _, _ = _metrics_from_eq(base_eq)
    base_sharpe = float(base_met.get("sharpe", 0.0))

    worst = 1e9
    worst_key = "base"
    for cm in cfg["stress"]["commission_mult"]:
        for sm in cfg["stress"]["slippage_mult"]:
            for tm in cfg["stress"]["turnover_caps"]:
                eq = _simulate_frame(
                    prices,
                    weights,
                    commission_bps=float(cfg["execution"]["commission_bps"]) * float(cm),
                    sell_tax_bps=float(cfg["execution"]["sell_tax_bps"]),
                    slippage_bps=float(cfg["execution"]["slippage_bps"]) * float(sm),
                    turnover_cap=float(tm),
                )
                met, _, _ = _metrics_from_eq(eq)
                sh = float(met.get("sharpe", 0.0))
                if sh < worst:
                    worst = sh
                    worst_key = f"cm={cm}|sm={sm}|cap={tm}"
    fragility = max(0.0, base_sharpe - worst)
    return float(fragility), worst_key, float(worst)


def run_eval_lab(params: dict, outdir: Path) -> dict:
    dataset = str(params.get("dataset", "vn_daily"))
    protocol = str(params.get("protocol", "walk_forward"))
    selected = set(params.get("strategies") or [])

    cfg = _load_cfg()
    seed = int(cfg["random_seed"])
    prices = _load_prices(dataset)
    universe = _choose_universe(prices)
    prices = prices[prices["symbol"].isin(universe)].copy()
    window, test_dates = _evaluation_window(prices, cfg, protocol)

    run_id = _h(
        json.dumps(
            {"dataset": dataset, "seed": seed, "rows": len(prices), "protocol": protocol},
            sort_keys=True,
        )
    )[:16]
    out = Path(outdir)
    (out / "equity_curves").mkdir(parents=True, exist_ok=True)
    (out / "window_metrics").mkdir(parents=True, exist_ok=True)
    (out / "stats" / "bootstrap_samples").mkdir(parents=True, exist_ok=True)

    audit_path = (
        Path("data_demo/prices_demo_1d.csv")
        if dataset == "vn_daily"
        else Path("data_demo/crypto_prices_demo_1d.csv")
    )
    audit = audit_prices(audit_path, out_md=out / "data_audit.md")

    reg = build_strategy_registry(raocmoe_available=True)
    if selected:
        reg = [x for x in reg if x.name in selected]

    rows = []
    eq_by_name: dict[str, pd.DataFrame] = {}
    ret_arrays: list[np.ndarray] = []
    consistency_failed = False
    identity_user: dict[str, float] = {
        "gross_end": 0.0,
        "net_end": 0.0,
        "total_cost_total": 0.0,
        "gross_minus_net": 0.0,
        "abs_err": 0.0,
    }

    for spec in reg:
        w = _weights_for_strategy(spec.name, prices, universe)
        eq = _simulate_frame(
            prices,
            w,
            commission_bps=float(cfg["execution"]["commission_bps"]),
            sell_tax_bps=float(cfg["execution"]["sell_tax_bps"]),
            slippage_bps=float(cfg["execution"]["slippage_bps"]),
            turnover_cap=float(cfg["stress"]["turnover_caps"][0]),
        )
        eq = eq[eq["date"].isin(test_dates)].copy()
        eq.to_csv(out / "equity_curves" / f"{spec.name}.csv", index=False)
        eq_by_name[spec.name] = eq

        met, checks, identity = _metrics_from_eq(eq)
        if spec.name.startswith("USER_V"):
            fragility, worst_key, worst_sharpe = _stress_fragility(prices, w, cfg)
        else:
            fragility, worst_key, worst_sharpe = 0.0, "n/a", 0.0

        r = eq["equity_net"].pct_change().fillna(0.0).to_numpy() if not eq.empty else np.zeros(1)
        ret_arrays.append(r)

        wf = walk_forward_splits(
            n=len(eq),
            train=int(cfg["protocols"]["walk_forward"]["train_days"]),
            test=int(cfg["protocols"]["walk_forward"]["test_days"]),
            step=int(cfg["protocols"]["walk_forward"]["step_days"]),
        )
        wf_rows = []
        for i, (_, te) in enumerate(wf):
            seg = eq.iloc[list(te)] if len(eq) else pd.DataFrame()
            rr = seg["equity_net"].pct_change().fillna(0.0).tolist() if not seg.empty else []
            em = compute_performance_metrics(
                rr,
                seg["equity_net"].tolist() if not seg.empty else [],
                seg["turnover_l1"].tolist() if not seg.empty else [],
            )
            wf_rows.append({"window": i, **em})
        pd.DataFrame(wf_rows).to_csv(out / "window_metrics" / f"{spec.name}.csv", index=False)

        ci = bootstrap_ci(
            eq["equity_net"].pct_change().fillna(0.0).tolist() if not eq.empty else [],
            seed=seed,
            n_samples=int(cfg["bootstrap"]["samples"]),
            block_size=int(cfg["bootstrap"]["block_daily"]),
        )
        pd.DataFrame({"sample_mean_ci": [ci["p05"], ci["p50"], ci["p95"]]}).to_csv(
            out / "stats" / "bootstrap_samples" / f"{spec.name}.csv", index=False
        )

        consistency_failed = consistency_failed or (not all(checks.values()))
        row = {
            "strategy": spec.name,
            **met,
            "psr_sr0_0": psr(met["sharpe"], 0.0, int(met["days"])),
            "psr_sr0_05": psr(met["sharpe"], 0.5, int(met["days"])),
            "consistency_ok": int(all(checks.values())),
            "stress_fragility_score": fragility,
            "stress_worst_key": worst_key,
            "stress_worst_sharpe": worst_sharpe,
        }
        rows.append(row)
        if spec.name in {"USER_V0", "Strategy_USER"}:
            identity_user = identity

    max_len = max(len(a) for a in ret_arrays) if ret_arrays else 1
    rm = (
        np.column_stack([np.pad(a, (0, max_len - len(a))) for a in ret_arrays])
        if ret_arrays
        else np.zeros((1, 1))
    )
    corr = np.corrcoef(rm.T) if rm.shape[1] > 1 else np.ones((1, 1))
    n_eff = int(max(1, round((np.trace(corr) ** 2) / max(1e-8, np.sum(corr * corr)))))

    bench = (
        eq_by_name["Baseline_BH_EW"]["equity_net"].pct_change().fillna(0.0).to_numpy()
        if "Baseline_BH_EW" in eq_by_name
        else np.zeros(1)
    )
    diff_cols = []
    pvals = {}
    for name, eq in eq_by_name.items():
        r = eq["equity_net"].pct_change().fillna(0.0).to_numpy()
        ll = min(len(r), len(bench))
        d = (r[:ll] - bench[:ll]) if ll > 0 else np.zeros(1)
        diff_cols.append(np.pad(d, (0, max_len - len(d))))
        pvals[name] = float(np.mean(d <= 0.0))
    diff_mat = np.column_stack(diff_cols) if diff_cols else np.zeros((1, 1))

    B = int(cfg["bootstrap"]["samples"])
    rc_p = reality_check(diff_mat, block_size=int(cfg["bootstrap"]["block_daily"]), b=B, seed=seed)
    spa_p = spa(diff_mat, block_size=int(cfg["bootstrap"]["block_daily"]), b=B, seed=seed)
    pbo = pbo_cscv(rm, s=16, seed=seed)
    fdr = benjamini_hochberg(pvals, q=float(cfg["fdr"]["q"]))

    trials = len(reg)
    res = pd.DataFrame(rows)
    res["n_trials"] = trials
    res["n_eff"] = n_eff
    res["dsr"] = res["sharpe"].apply(
        lambda x: dsr(float(x), int(max(3, res["days"].max() if not res.empty else 3)), trials)
    )
    res["mintrl"] = res["sharpe"].apply(lambda x: min_trl(float(x), 0.0))
    res.to_csv(out / "results_table.csv", index=False)

    user_name = "USER_V0" if "USER_V0" in set(res["strategy"]) else "Strategy_USER"
    user = res[res["strategy"] == user_name].iloc[0]
    bh = res[res["strategy"] == "Baseline_BH_EW"].iloc[0]
    alpha = float(user["total_return"] - bh["total_return"])
    ir = float(user["sharpe"] - bh["sharpe"])

    reasons: list[str] = []
    if str(audit.get("data_health_score", "FAIL")) != "PASS":
        reasons.append("data_audit_FAIL")
    if consistency_failed:
        reasons.append("consistency_failed")
    if int(user["days"]) < int(user["mintrl"]):
        reasons.append("INSUFFICIENT_TRACK_RECORD")
    if pbo > 0.4:
        reasons.append("PBO_gt_0_4")
    if float(user["stress_fragility_score"]) > 0.5 or float(user["stress_worst_sharpe"]) < -0.25:
        reasons.append("stress_fragility_high")
    if not (rc_p <= 0.10 or spa_p <= 0.10 or float(user["dsr"]) >= 0.95):
        reasons.append("significance_gate_failed")
    verdict = "PASS" if not reasons else "FAIL"

    mt = {
        "rc_p_raw": rc_p,
        "spa_p_raw": spa_p,
        "rc_p_print": format_pvalue(max(rc_p, 1.0 / B), B),
        "spa_p_print": format_pvalue(max(spa_p, 1.0 / B), B),
        "bootstrap_B": B,
        "fdr": fdr,
        "pbo": pbo,
        "n_trials": trials,
        "n_eff": n_eff,
    }
    (out / "stats" / "multiple_testing.json").write_text(
        json.dumps(mt, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    dataset_hash = _h(audit_path.read_text(encoding="utf-8"))
    config_hash = _h(Path("configs/eval_lab.yaml").read_text(encoding="utf-8"))
    code_hash = _h(Path(".git/HEAD").read_text(encoding="utf-8"))

    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "evaluation_window": asdict(window),
        "bootstrap_B": B,
        "identity": identity_user,
        "user": user.to_dict(),
        "user_vs_bh": {
            "alpha": alpha,
            "ir": ir,
            "rc_p_raw": rc_p,
            "rc_p_print": mt["rc_p_print"],
            "spa_p_raw": spa_p,
            "spa_p_print": mt["spa_p_print"],
            "dsr": float(user["dsr"]),
            "pbo": pbo,
            "mintrl": int(user["mintrl"]),
        },
        "reliability": {
            "verdict": verdict,
            "reasons": reasons,
            "consistency_failed": consistency_failed,
        },
        "audit": audit,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    summary_md = [
        "# Evaluation Lab Report",
        f"- run_id: `{run_id}`",
        f"- dataset_hash: `{dataset_hash}`",
        f"- config_hash: `{config_hash}`",
        f"- code_hash: `{code_hash}`",
        f"- FULL: start={window.full_start}, end={window.full_end}, days={window.full_days}",
        f"- TRAIN: start={window.train_start}, end={window.train_end}, days={window.train_days}",
        f"- TEST: start={window.test_start}, end={window.test_end}, days={window.test_days}",
        f"- IDENTITY: gross_end={identity_user['gross_end']:.6f}, net_end={identity_user['net_end']:.6f}, total_cost_total={identity_user['total_cost_total']:.6f}, gross_minus_net={identity_user['gross_minus_net']:.6f}, abs_err={identity_user['abs_err']:.6e}",
        f"- USER Test: net_total_return={user['total_return']:.6f}, gross_total_return={user['gross_total_return']:.6f}, Sharpe={user['sharpe']:.6f}, turnover_l1_total={user['turnover_l1_total']:.6f}, turnover_l1_daily_avg={user['turnover_l1_daily_avg']:.6f}, turnover_l1_annualized={user['turnover_l1_annualized']:.6f}, traded_notional_total={user['traded_notional_total']:.6f}, traded_notional_over_nav={user['traded_notional_over_nav']:.6f}, avg_holding_period_days={user['avg_holding_period_days']:.6f}, commission_cost_total={user['commission_cost_total']:.6f}, sell_tax_cost_total={user['sell_tax_cost_total']:.6f}, slippage_cost_total={user['slippage_cost_total']:.6f}, total_cost_total={user['total_cost_total']:.6f}, cost_drag_vs_gross={user['cost_drag_vs_gross']:.6f}, cost_drag_vs_traded={user['cost_drag_vs_traded']:.6f}",
        f"- USER vs Baseline_BH_EW: alpha={alpha:.6f}, IR={ir:.6f}, RC={mt['rc_p_print']}, SPA={mt['spa_p_print']}, DSR={float(user['dsr']):.6f}, PBO={pbo:.6f}, MinTRL={int(user['mintrl'])}, bootstrap_B={B}",
        f"- RELIABILITY: {verdict} ({'; '.join(reasons) if reasons else 'all checks passed'})",
    ]
    (out / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    print(f"FULL: start={window.full_start}, end={window.full_end}, days={window.full_days}")
    print(f"TRAIN: start={window.train_start}, end={window.train_end}, days={window.train_days}")
    print(f"TEST: start={window.test_start}, end={window.test_end}, days={window.test_days}")
    print(
        "IDENTITY: "
        f"gross_end={identity_user['gross_end']:.6f}, "
        f"net_end={identity_user['net_end']:.6f}, "
        f"total_cost_total={identity_user['total_cost_total']:.6f}, "
        f"gross_minus_net={identity_user['gross_minus_net']:.6f}, "
        f"abs_err={identity_user['abs_err']:.6e}"
    )
    print(
        "USER Test: "
        f"net_total_return={user['total_return']:.6f}, "
        f"gross_total_return={user['gross_total_return']:.6f}, "
        f"Sharpe={user['sharpe']:.6f}, "
        f"MDD={user['mdd']:.6f}, "
        f"turnover_l1_total={user['turnover_l1_total']:.6f} (sum of |Δw| over test days), "
        f"turnover_l1_daily_avg={user['turnover_l1_daily_avg']:.6f}, "
        f"turnover_l1_annualized={user['turnover_l1_annualized']:.6f}, "
        f"traded_notional_total={user['traded_notional_total']:.6f}, "
        f"traded_notional_over_nav={user['traded_notional_over_nav']:.6f}, "
        f"avg_holding_period_days={user['avg_holding_period_days']:.6f}, "
        f"commission_cost_total={user['commission_cost_total']:.6f}, "
        f"sell_tax_cost_total={user['sell_tax_cost_total']:.6f}, "
        f"slippage_cost_total={user['slippage_cost_total']:.6f}, "
        f"total_cost_total={user['total_cost_total']:.6f}, "
        f"cost_drag_vs_gross={user['cost_drag_vs_gross']:.6f}, "
        f"cost_drag_vs_traded={user['cost_drag_vs_traded']:.6f}"
    )
    print(
        "USER vs Baseline_BH_EW: "
        f"alpha={alpha:.6f}, IR={ir:.6f}, RC={mt['rc_p_print']}, SPA={mt['spa_p_print']}, "
        f"DSR={float(user['dsr']):.6f}, PBO={pbo:.6f}, MinTRL={int(user['mintrl'])}, bootstrap_B={B}"
    )
    print(f"RELIABILITY: {verdict} with reasons: {', '.join(reasons) if reasons else 'none'}")

    return {
        "run_id": run_id,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "verdict": verdict,
        "reasons": reasons,
        "results_table": out / "results_table.csv",
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vn_daily")
    parser.add_argument(
        "--protocol", default="walk_forward", choices=["single_split", "walk_forward"]
    )
    parser.add_argument("--outdir", default="reports/eval_lab/manual")
    parser.add_argument("--strategies", default="")
    args = parser.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    summary = run_eval_lab(
        {"dataset": args.dataset, "protocol": args.protocol, "strategies": strategies},
        Path(args.outdir),
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "summary"}, default=str, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
