from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.eval_lab import (
    benjamini_hochberg,
    bootstrap_ci,
    build_strategy_registry,
    check_cost_nonnegativity,
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
    user_strategy_adapter,
)
from scripts.audit_dataset_prices import audit_prices


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
    if not path.exists():
        raise FileNotFoundError(str(path))
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
    if name == "Strategy_USER":
        return user_strategy_adapter.generate_weights(frame, universe)
    return raocmoe_adapter.generate_weights(frame, universe, name)


def _simulate(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    commission_bps: float,
    sell_tax_bps: float,
    slippage_bps: float,
    turnover_cap: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, bool]]:
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

    turnover_total = 0.0
    traded_notional_total = 0.0
    nav_sum = 0.0
    commission_total = 0.0
    sell_tax_total = 0.0
    slippage_total = 0.0
    cum_cost = 0.0

    dates = sorted(f["date"].unique())
    for d in dates[:-1]:
        day = f[f["date"] == d]
        if day.empty:
            continue
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
        cum_cost = gross_eq - net_eq

        turnover_total += l1
        traded_notional_total += l1 * net_eq
        nav_sum += net_eq
        commission_total += commission
        sell_tax_total += sell_tax
        slippage_total += slippage

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
                "cum_cost": cum_cost,
            }
        )
        prev = target

    eq = pd.DataFrame(rows)
    net_r = eq["equity_net"].pct_change().fillna(0.0).tolist() if not eq.empty else []
    perf = compute_performance_metrics(
        net_r,
        eq["equity_net"].tolist() if not eq.empty else [],
        eq["turnover_l1"].tolist() if not eq.empty else [],
    )
    perf.update(compute_tail_metrics(net_r))

    days = float(len(eq))
    avg_nav = float(nav_sum / max(1.0, days))
    total_cost_total = commission_total + sell_tax_total + slippage_total
    gross_end = float(eq["equity_gross"].iloc[-1]) if not eq.empty else 1.0
    perf.update(
        {
            "days": days,
            "turnover_l1_total": turnover_total,
            "turnover_l1_daily_avg": turnover_total / max(1.0, days),
            "turnover_l1_annualized": (turnover_total / max(1.0, days)) * 252.0,
            "traded_notional_total": traded_notional_total,
            "avg_nav": avg_nav,
            "traded_notional_over_nav": traded_notional_total / max(1e-8, avg_nav),
            "commission_cost_total": commission_total,
            "sell_tax_cost_total": sell_tax_total,
            "slippage_cost_total": slippage_total,
            "total_cost_total": total_cost_total,
            "cost_drag_vs_gross": total_cost_total / max(1e-8, gross_end),
            "cost_drag_vs_traded": total_cost_total / max(1e-8, traded_notional_total),
            "gross_total_return": gross_end - 1.0,
        }
    )

    consistency = {
        "equity_identity": check_equity_identity(
            eq["equity_gross"].tolist() if not eq.empty else [],
            eq["equity_net"].tolist() if not eq.empty else [],
            eq["cum_cost"].tolist() if not eq.empty else [],
        ),
        "return_identity": check_return_identity(
            eq["equity_net"].tolist() if not eq.empty else [],
            perf["total_return"],
        ),
        "cost_nonnegative": check_cost_nonnegativity(
            {
                "commission": commission_total,
                "sell_tax": sell_tax_total,
                "slippage": slippage_total,
                "total_cost": total_cost_total,
            }
        ),
    }
    return eq, perf, consistency


def _stress_fragility(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    base_net_return: float,
    base_sharpe: float,
    cfg: dict,
) -> float:
    _, m2, _ = _simulate(
        prices,
        weights,
        commission_bps=float(cfg["execution"]["commission_bps"]) * 2.0,
        sell_tax_bps=float(cfg["execution"]["sell_tax_bps"]),
        slippage_bps=float(cfg["execution"]["slippage_bps"]) * 2.0,
        turnover_cap=float(cfg["stress"]["turnover_caps"][0]),
    )
    drop = base_net_return - float(m2["total_return"])
    sharpe_flip = (
        1.0
        if (base_sharpe > 0 and m2["sharpe"] < 0 and (base_sharpe - m2["sharpe"]) > 0.5)
        else 0.0
    )
    frag = max(0.0, drop - 0.10) + sharpe_flip
    return float(frag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vn_daily")
    args = parser.parse_args()

    cfg = _load_cfg()
    seed = int(cfg["random_seed"])
    prices = _load_prices(args.dataset)
    universe = _choose_universe(prices)
    prices = prices[prices["symbol"].isin(universe)].copy()

    run_id = _h(
        json.dumps({"dataset": args.dataset, "seed": seed, "rows": len(prices)}, sort_keys=True)
    )[:16]
    out = Path("reports") / "eval_lab" / run_id
    (out / "equity_curves").mkdir(parents=True, exist_ok=True)
    (out / "window_metrics").mkdir(parents=True, exist_ok=True)
    (out / "stats" / "bootstrap_samples").mkdir(parents=True, exist_ok=True)

    audit_path = (
        Path("data_demo/prices_demo_1d.csv")
        if args.dataset == "vn_daily"
        else Path("data_demo/crypto_prices_demo_1d.csv")
    )
    audit = audit_prices(audit_path, out_md=out / "data_audit.md")

    reg = build_strategy_registry(raocmoe_available=True)
    stress_variants = (
        len(cfg["stress"]["commission_mult"])
        * len(cfg["stress"]["slippage_mult"])
        * len(cfg["stress"]["liquidity_mult"])
        * len(cfg["stress"]["turnover_caps"])
        * len(cfg["stress"]["limit_penalty"])
    )

    rows = []
    eq_by_name: dict[str, pd.DataFrame] = {}
    ret_arrays: list[np.ndarray] = []
    consistency_failed = False

    for spec in reg:
        w = _weights_for_strategy(spec.name, prices, universe)
        eq, met, checks = _simulate(
            prices,
            w,
            commission_bps=float(cfg["execution"]["commission_bps"]),
            sell_tax_bps=float(cfg["execution"]["sell_tax_bps"]),
            slippage_bps=float(cfg["execution"]["slippage_bps"]),
            turnover_cap=float(cfg["stress"]["turnover_caps"][0]),
        )
        eq.to_csv(out / "equity_curves" / f"{spec.name}.csv", index=False)
        eq_by_name[spec.name] = eq
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
        rows.append(
            {
                "strategy": spec.name,
                **met,
                "psr_sr0_0": psr(met["sharpe"], 0.0, int(met["days"])),
                "psr_sr0_05": psr(met["sharpe"], 0.5, int(met["days"])),
                "consistency_ok": int(all(checks.values())),
            }
        )

    max_len = max(len(a) for a in ret_arrays)
    rm = np.column_stack([np.pad(a, (0, max_len - len(a))) for a in ret_arrays])
    corr = np.corrcoef(rm.T) if rm.shape[1] > 1 else np.ones((1, 1))
    n_eff = int(max(1, round((np.trace(corr) ** 2) / max(1e-8, np.sum(corr * corr)))))

    bench = eq_by_name["Baseline_BH_EW"]["equity_net"].pct_change().fillna(0.0).to_numpy()
    diff_cols = []
    pvals = {}
    for name, eq in eq_by_name.items():
        r = eq["equity_net"].pct_change().fillna(0.0).to_numpy()
        ll = min(len(r), len(bench))
        d = (r[:ll] - bench[:ll]) if ll > 0 else np.zeros(1)
        diff_cols.append(np.pad(d, (0, max_len - len(d))))
        pvals[name] = float(np.mean(d <= 0.0))
    diff_mat = np.column_stack(diff_cols)
    B = int(cfg["bootstrap"]["samples"])
    rc_p = reality_check(diff_mat, block_size=int(cfg["bootstrap"]["block_daily"]), b=B, seed=seed)
    spa_p = spa(diff_mat, block_size=int(cfg["bootstrap"]["block_daily"]), b=B, seed=seed)
    assert 0.0 <= rc_p <= 1.0 and 0.0 <= spa_p <= 1.0
    if B > 0:
        assert rc_p >= (1.0 / B) and spa_p >= (1.0 / B)
    fdr = benjamini_hochberg(pvals, q=float(cfg["fdr"]["q"]))
    pbo = pbo_cscv(rm, s=16, seed=seed)

    trials = len(reg) + stress_variants
    res = pd.DataFrame(rows)
    res["n_trials"] = trials
    res["n_eff"] = n_eff
    res["dsr"] = res["sharpe"].apply(
        lambda x: dsr(float(x), int(max(3, res["days"].max())), trials)
    )
    res["mintrl"] = res["sharpe"].apply(lambda x: min_trl(float(x), 0.0))
    res.to_csv(out / "results_table.csv", index=False)

    user = res[res["strategy"] == "Strategy_USER"].iloc[0]
    bh = res[res["strategy"] == "Baseline_BH_EW"].iloc[0]
    alpha = float(user["total_return"] - bh["total_return"])
    ir = float(user["sharpe"] - bh["sharpe"])

    user_weights = _weights_for_strategy("Strategy_USER", prices, universe)
    fragility = _stress_fragility(
        prices, user_weights, float(user["total_return"]), float(user["sharpe"]), cfg
    )

    reasons = []
    if str(audit.get("data_health_score", "FAIL")) != "PASS":
        reasons.append("data_audit_FAIL")
    if consistency_failed:
        reasons.append("consistency_failed")
    if int(user["days"]) < int(user["mintrl"]):
        reasons.append("INSUFFICIENT_TRACK_RECORD")
    if pbo > 0.4:
        reasons.append("PBO_gt_0_4")
    if fragility > 0.0:
        reasons.append("stress_fragility_high")
    verdict = "PASS" if not reasons else "FAIL"

    mt = {
        "rc_p_raw": rc_p,
        "spa_p_raw": spa_p,
        "rc_p_print": format_pvalue(rc_p, B),
        "spa_p_print": format_pvalue(spa_p, B),
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
        "dataset": args.dataset,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "date_min": str(prices["date"].min().date()),
        "date_max": str(prices["date"].max().date()),
        "user": user.to_dict(),
        "user_vs_bh": {
            "alpha": alpha,
            "ir": ir,
            "rc_p_raw": rc_p,
            "rc_p_print": format_pvalue(rc_p, B),
            "dsr": float(user["dsr"]),
            "pbo": pbo,
            "mintrl": int(user["mintrl"]),
        },
        "n_trials": trials,
        "n_eff": n_eff,
        "stress_fragility_score": fragility,
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
        f"- date_range: {summary['date_min']} -> {summary['date_max']}",
        f"- USER test: net_total_return={user['total_return']:.6f}, gross_total_return={user['gross_total_return']:.6f}, Sharpe={user['sharpe']:.6f}, turnover_l1_total={user['turnover_l1_total']:.6f}, turnover_l1_daily_avg={user['turnover_l1_daily_avg']:.6f}, turnover_l1_annualized={user['turnover_l1_annualized']:.6f}, cost_drag_vs_gross={user['cost_drag_vs_gross']:.6f}, cost_drag_vs_traded={user['cost_drag_vs_traded']:.6f}",
        f"- USER vs Baseline_BH_EW: alpha={alpha:.6f}, IR={ir:.6f}, RC={format_pvalue(rc_p, B)}, DSR={float(user['dsr']):.6f}, PBO={pbo:.6f}, MinTRL={int(user['mintrl'])}",
        f"- RELIABILITY: {verdict} ({'; '.join(reasons) if reasons else 'all checks passed'})",
    ]
    (out / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    print("python scripts/run_eval_lab.py --dataset vn_daily")
    print(
        f"USER Test: net_total_return={user['total_return']:.6f}, gross_total_return={user['gross_total_return']:.6f}, CAGR={user['cagr']:.6f}, MDD={user['mdd']:.6f}, Sharpe={user['sharpe']:.6f}, turnover_l1_total={user['turnover_l1_total']:.6f}, cost_drag_vs_gross={user['cost_drag_vs_gross']:.6f}, cost_drag_vs_traded={user['cost_drag_vs_traded']:.6f}, days={int(user['days'])}"
    )
    print(
        f"USER vs Baseline_BH_EW: alpha={alpha:.6f}, IR={ir:.6f}, RC={format_pvalue(rc_p, B)}, DSR={float(user['dsr']):.6f}, PBO={pbo:.6f}, MinTRL={int(user['mintrl'])}"
    )
    print(f"RELIABILITY: {verdict} with reasons: {', '.join(reasons) if reasons else 'none'}")
    print(f"REPORT_PATH: {out}")
    for rel in [
        "summary.md",
        "summary.json",
        "results_table.csv",
        "stats/multiple_testing.json",
        "data_audit.md",
    ]:
        print(f" - {out / rel}")


if __name__ == "__main__":
    main()
