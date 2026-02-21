from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIXED_ORDER = [
    "Baseline_BH_EW",
    "USER_V0",
    "USER_V1_STABILITY",
    "USER_V2_COSTAWARE",
    "USER_V3_REGIME_UQ",
    "RAOCMOE_FULL",
]


def _latest_run(base: Path, must_contain: str) -> Path | None:
    runs = [p for p in base.iterdir() if p.is_dir() and (p / must_contain).exists()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "strategy",
        "total_return",
        "sharpe",
        "mdd",
        "turnover_l1_annualized",
        "cost_drag_vs_traded",
        "stress_fragility_score",
        "variant_pass",
        "objective_score",
    ]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append("NA" if pd.isna(v) else f"{v:.6f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _print_eval_summary(summary: dict, table: pd.DataFrame) -> None:
    print("EVAL_LAB CHAT SUMMARY")
    print(
        f"run_id={summary['run_id']} dataset={summary['dataset']} verdict={summary['reliability']['verdict']}"
    )
    w = summary["evaluation_window"]
    print(
        f"FULL[{w['full_start']}..{w['full_end']}] days={w['full_days']} | "
        f"TRAIN[{w['train_start']}..{w['train_end']}] days={w['train_days']} | "
        f"TEST[{w['test_start']}..{w['test_end']}] days={w['test_days']}"
    )
    ident = summary["identity"]
    print(
        "IDENTITY "
        f"gross_end={ident['gross_end']:.6f} net_end={ident['net_end']:.6f} "
        f"total_cost_total={ident['total_cost_total']:.6f} abs_err={ident['abs_err']:.6e}"
    )

    ordered = table.copy()
    ordered["_ord"] = ordered["strategy"].apply(
        lambda x: FIXED_ORDER.index(x) if x in FIXED_ORDER else len(FIXED_ORDER)
    )
    ordered = ordered.sort_values(["_ord", "strategy"]).drop(columns=["_ord"])
    print(_markdown_table(ordered))

    base_row = table[table["strategy"] == "USER_V0"]
    if not base_row.empty:
        b = base_row.iloc[0]
        print("DELTA vs USER_V0")
        for _, r in table[table["strategy"].astype(str).str.startswith("USER_V")].iterrows():
            if r["strategy"] == "USER_V0":
                continue
            print(
                f"{r['strategy']}: ΔSharpe={(float(r['sharpe'])-float(b['sharpe'])):.6f}, "
                f"ΔTurn_ann={(float(r['turnover_l1_annualized'])-float(b['turnover_l1_annualized'])):.6f}, "
                f"ΔMDD={(float(r['mdd'])-float(b['mdd'])):.6f}"
            )

    print(f"CHOSEN_DEFAULT={summary.get('chosen_default', 'USER_V0')}")
    print(f"WHY_CHOSEN={summary.get('why_chosen', 'n/a')}")
    print("reasons=" + (", ".join(summary["reliability"]["reasons"]) or "none"))


def _print_lockbox_block() -> None:
    improvements = _latest_run(Path("reports/improvements"), "improvement_summary.json")
    if improvements is None:
        return
    payload = json.loads((improvements / "improvement_summary.json").read_text(encoding="utf-8"))
    print("LOCKBOX VERIFY SUMMARY")
    print(f"run_id={payload['run_id']} dataset={payload['dataset']}")
    print(
        f"DEV_WINNER={payload['dev_winner']} LOCKBOX_VERDICT={payload['lockbox_verdict']} "
        f"FINAL_DEFAULT={payload['final_default']}"
    )
    if "dev_scoreboard" in payload and "lockbox_scoreboard" in payload:
        print(
            f"artifacts dev_scoreboard={payload['dev_scoreboard']} "
            f"lockbox_scoreboard={payload['lockbox_scoreboard']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    base = Path("reports/eval_lab")
    run_dir = (base / args.run_id) if args.run_id else _latest_run(base, "summary.json")
    if run_dir is None:
        raise FileNotFoundError("No eval_lab run with summary.json found")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(run_dir / "results_table.csv")

    _print_eval_summary(summary, table)
    _print_lockbox_block()


if __name__ == "__main__":
    main()
