from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _latest_run(base: Path) -> Path:
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "model_outputs").exists()]
    if not runs:
        raise FileNotFoundError("No eval_lab run with model_outputs found")
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]


def _print_strategy(run_dir: Path, strategy: str, days: int, topk: int) -> None:
    fp = run_dir / "model_outputs" / f"{strategy}.csv"
    if not fp.exists():
        print(f"{strategy}: missing")
        return
    df = pd.read_csv(fp)
    if df.empty:
        print(f"{strategy}: empty")
        return
    last_days = sorted(df["decision_date"].astype(str).unique())[-days:]
    snap = df[df["decision_date"].astype(str).isin(last_days)].copy()
    snap["abs_target_w"] = snap["target_w"].abs()
    snap = (
        snap.sort_values(["decision_date", "abs_target_w"], ascending=[False, False])
        .groupby("decision_date")
        .head(topk)
    )
    cols = [
        "decision_date",
        "symbol",
        "score_raw",
        "regime",
        "prev_w",
        "target_w",
        "delta_w",
        "side",
        "est_cost_bps",
        "realized_return_next",
        "notes",
    ]
    print(f"\n{strategy}")
    print(snap[cols].to_string(index=False))
    warns = []
    if (snap["delta_w"].abs() > 0.5).any():
        warns.append("large_delta_w")
    if (snap["est_cost_bps"] > 50).any():
        warns.append("high_est_cost_bps")
    print("warnings=" + (", ".join(warns) if warns else "none"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    base = Path("reports/eval_lab")
    run_dir = base / args.run_id if args.run_id else _latest_run(base)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    chosen_default = str(summary.get("chosen_default", "USER_V0"))

    print("MODEL OUTPUT SNAPSHOT")
    print(f"run_id={run_dir.name}")
    _print_strategy(run_dir, "USER_V0", args.days, args.topk)
    _print_strategy(run_dir, chosen_default, args.days, args.topk)


if __name__ == "__main__":
    main()
