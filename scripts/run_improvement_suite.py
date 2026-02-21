from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.run_eval_lab import run_eval_lab


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vn_daily")
    args = parser.parse_args()

    strategies = [
        "Baseline_BH_EW",
        "USER_V0",
        "USER_V1_STABILITY",
        "USER_V2_COSTAWARE",
        "USER_V3_REGIME_UQ",
        "RAOCMOE_FULL",
    ]

    run_id = _h(json.dumps({"dataset": args.dataset, "strategies": strategies}, sort_keys=True))[
        :16
    ]
    out = Path("reports/improvements") / run_id
    out.mkdir(parents=True, exist_ok=True)

    result = run_eval_lab(
        {"dataset": args.dataset, "protocol": "walk_forward", "strategies": strategies},
        outdir=out / "eval_lab",
    )
    summary = result["summary"]
    table = pd.read_csv(result["results_table"])

    for name in [
        "Baseline_BH_EW",
        "USER_V0",
        "USER_V1_STABILITY",
        "USER_V2_COSTAWARE",
        "USER_V3_REGIME_UQ",
        "RAOCMOE_FULL",
    ]:
        row = table[table["strategy"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        verdict = (
            "PASS"
            if (
                int(r.get("consistency_ok", 0)) == 1
                and float(r.get("stress_fragility_score", 1.0)) <= 0.5
            )
            else "FAIL"
        )
        print(
            f"[{name}] Test net_ret={float(r['total_return']):.6f}, Sharpe={float(r['sharpe']):.6f}, "
            f"MDD={float(r['mdd']):.6f}, Turn_ann={float(r['turnover_l1_annualized']):.6f}, "
            f"CostDrag_traded={float(r['cost_drag_vs_traded']):.6f}, Fragility={float(r['stress_fragility_score']):.6f}, Verdict={verdict}"
        )

    base = table[table["strategy"] == "USER_V0"].iloc[0]
    deltas = {}
    for nm in ["USER_V1_STABILITY", "USER_V2_COSTAWARE", "USER_V3_REGIME_UQ"]:
        row = table[table["strategy"] == nm].iloc[0]
        deltas[nm] = {
            "d_sharpe": float(row["sharpe"] - base["sharpe"]),
            "d_turn_ann": float(row["turnover_l1_annualized"] - base["turnover_l1_annualized"]),
            "d_cost_drag_traded": float(row["cost_drag_vs_traded"] - base["cost_drag_vs_traded"]),
            "d_fragility": float(row["stress_fragility_score"] - base["stress_fragility_score"]),
        }

    print("DELTA vs USER_V0")
    for k, v in deltas.items():
        print(
            f"{k}: ΔSharpe={v['d_sharpe']:.6f}, ΔTurn_ann={v['d_turn_ann']:.6f}, "
            f"ΔCostDrag_traded={v['d_cost_drag_traded']:.6f}, ΔFragility={v['d_fragility']:.6f}"
        )

    crit = {
        "turnover_reduction": float(base["turnover_l1_annualized"] * 0.5),
        "costdrag_reduction": float(base["cost_drag_vs_traded"] * 0.7),
        "fragility_reduction": float(base["stress_fragility_score"] - 0.25),
        "max_sharpe_drop": float(base["sharpe"] - 0.15),
    }

    score = []
    for nm in ["USER_V1_STABILITY", "USER_V2_COSTAWARE", "USER_V3_REGIME_UQ"]:
        row = table[table["strategy"] == nm].iloc[0]
        checks = {
            "turnover_reduction": float(row["turnover_l1_annualized"])
            <= crit["turnover_reduction"],
            "costdrag_reduction": float(row["cost_drag_vs_traded"]) <= crit["costdrag_reduction"],
            "fragility_reduction": float(row["stress_fragility_score"])
            <= crit["fragility_reduction"],
            "sharpe_guard": float(row["sharpe"]) >= crit["max_sharpe_drop"],
        }
        score.append({"strategy": nm, **checks})

    candidates = table[
        table["strategy"].isin(["USER_V1_STABILITY", "USER_V2_COSTAWARE", "USER_V3_REGIME_UQ"])
    ].copy()
    candidates["pass_reliability"] = (candidates["stress_fragility_score"] <= 0.5) & (
        candidates["consistency_ok"] == 1
    )
    passing = candidates[candidates["pass_reliability"]]
    if not passing.empty:
        chosen = passing.sort_values(
            ["stress_fragility_score", "sharpe", "turnover_l1_annualized"],
            ascending=[True, False, True],
        ).iloc[0]["strategy"]
    else:
        fallback = candidates[candidates["total_return"] >= -0.02]
        if fallback.empty:
            fallback = candidates
        chosen = fallback.sort_values(
            ["stress_fragility_score", "turnover_l1_annualized"], ascending=[True, True]
        ).iloc[0]["strategy"]
    print(f"CHOSEN_DEFAULT={chosen}")

    report = {
        "run_id": run_id,
        "dataset": args.dataset,
        "dataset_hash": summary["dataset_hash"],
        "config_hash": summary["config_hash"],
        "code_hash": summary["code_hash"],
        "deltas": deltas,
        "criteria": score,
        "chosen_default": chosen,
    }
    (out / "improvement_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Improvement Summary",
        f"- run_id: `{run_id}`",
        f"- dataset_hash: `{summary['dataset_hash']}`",
        f"- config_hash: `{summary['config_hash']}`",
        f"- code_hash: `{summary['code_hash']}`",
        "## Delta vs USER_V0",
    ]
    for k, v in deltas.items():
        lines.append(
            f"- {k}: ΔSharpe={v['d_sharpe']:.6f}, ΔTurn_ann={v['d_turn_ann']:.6f}, ΔCostDrag_traded={v['d_cost_drag_traded']:.6f}, ΔFragility={v['d_fragility']:.6f}"
        )
    lines.append("## Acceptance Criteria")
    for row in score:
        lines.append(
            f"- {row['strategy']}: turnover_reduction={row['turnover_reduction']}, costdrag_reduction={row['costdrag_reduction']}, fragility_reduction={row['fragility_reduction']}, sharpe_guard={row['sharpe_guard']}"
        )
    lines.append(f"- CHOSEN_DEFAULT={chosen}")
    (out / "improvement_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
