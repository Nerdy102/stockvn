from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from scripts.run_eval_lab import run_eval_lab


OBJECTIVE_WEIGHTS = {
    "sharpe": 1.0,
    "fragility": -0.5,
    "turnover": -0.05,
    "mdd_abs": -0.2,
}


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _fmt(v: float) -> str:
    return "NA" if math.isnan(v) else f"{v:.6f}"


def objective_score(row: pd.Series) -> float:
    return (
        OBJECTIVE_WEIGHTS["sharpe"] * float(row["sharpe"])
        + OBJECTIVE_WEIGHTS["fragility"] * float(row["stress_fragility_score"])
        + OBJECTIVE_WEIGHTS["turnover"] * float(row["turnover_l1_annualized"])
        + OBJECTIVE_WEIGHTS["mdd_abs"] * abs(float(row["mdd"]))
    )


def choose_dev_winner(dev_t: pd.DataFrame) -> str:
    pool = dev_t[
        (dev_t["strategy"].astype(str).str.startswith("USER_V")) & (dev_t["variant_pass"] == 1)
    ]
    if pool.empty:
        return "NONE"
    ranked = pool.sort_values(["objective_score", "strategy"], ascending=[False, True])
    return str(ranked.iloc[0]["strategy"])


def _date_bounds(dataset: str) -> tuple[str, str, str, str]:
    src = (
        "data_demo/prices_demo_1d.csv"
        if dataset == "vn_daily"
        else "data_demo/crypto_prices_demo_1d.csv"
    )
    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].dt.date.unique().tolist())[:-1]
    split = max(1, int(len(dates) * 0.8))
    dev = dates[:split]
    lock = dates[split:] or [dates[-1]]
    return str(dev[0]), str(dev[-1]), str(lock[0]), str(lock[-1])


def _one_line(prefix: str, row: pd.Series, verified: str = "") -> str:
    suffix = f", Verified={verified}" if verified else ""
    verdict = "PASS" if int(row.get("variant_pass", 0)) == 1 else "FAIL"
    return (
        f"[{row['strategy']}] {prefix} net_ret={float(row['total_return']):.6f}, "
        f"Sharpe={float(row['sharpe']):.6f}, MDD={float(row['mdd']):.6f}, "
        f"Turn_ann={float(row['turnover_l1_annualized']):.6f}, "
        f"CostDrag_traded={_fmt(float(row['cost_drag_vs_traded']))}, "
        f"Fragility={float(row['stress_fragility_score']):.6f}, Verdict={verdict}{suffix}"
    )


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

    dev_start, dev_end, lock_start, lock_end = _date_bounds(args.dataset)
    dev = run_eval_lab(
        {
            "dataset": args.dataset,
            "protocol": "walk_forward",
            "strategies": strategies,
            "date_start": dev_start,
            "date_end": dev_end,
        },
        outdir=out / "eval_dev",
    )
    lock = run_eval_lab(
        {
            "dataset": args.dataset,
            "protocol": "walk_forward",
            "strategies": strategies,
            "date_start": lock_start,
            "date_end": lock_end,
        },
        outdir=out / "eval_lockbox",
    )

    dev_t = pd.read_csv(dev["results_table"])
    lock_t = pd.read_csv(lock["results_table"])
    dev_t["objective_score"] = dev_t.apply(objective_score, axis=1)
    lock_t["objective_score"] = lock_t.apply(objective_score, axis=1)

    dev_t.to_csv(out / "dev_scoreboard.csv", index=False)
    lock_t.to_csv(out / "lockbox_scoreboard.csv", index=False)

    dev_winner = choose_dev_winner(dev_t)
    lock_row = lock_t[lock_t["strategy"] == dev_winner]
    lockbox_verified = bool(
        (not lock_row.empty) and int(lock_row.iloc[0].get("variant_pass", 0)) == 1
    )

    for _, row in dev_t.iterrows():
        print(_one_line("DEV", row))
    for _, row in lock_t.iterrows():
        verified = "YES" if (row["strategy"] == dev_winner and lockbox_verified) else "NO"
        print(_one_line("LOCKBOX", row, verified=verified if row["strategy"] == dev_winner else ""))

    base = dev_t[dev_t["strategy"] == "USER_V0"].iloc[0]
    print("DELTA vs USER_V0")
    for _, row in dev_t[dev_t["strategy"].astype(str).str.startswith("USER_V")].iterrows():
        if row["strategy"] == "USER_V0":
            continue
        print(
            f"{row['strategy']}: ΔSharpe={float(row['sharpe']-base['sharpe']):.6f}, "
            f"ΔTurn_ann={float(row['turnover_l1_annualized']-base['turnover_l1_annualized']):.6f}, "
            f"ΔFragility={float(row['stress_fragility_score']-base['stress_fragility_score']):.6f}"
        )

    if dev_winner != "NONE":
        w = dev_t[dev_t["strategy"] == dev_winner].iloc[0]
        print("DELTA vs DEV winner")
        for _, row in dev_t[dev_t["strategy"].astype(str).str.startswith("USER_V")].iterrows():
            if row["strategy"] == dev_winner:
                continue
            print(
                f"{row['strategy']}: ΔSharpe={float(row['sharpe']-w['sharpe']):.6f}, "
                f"ΔTurn_ann={float(row['turnover_l1_annualized']-w['turnover_l1_annualized']):.6f}, "
                f"ΔFragility={float(row['stress_fragility_score']-w['stress_fragility_score']):.6f}"
            )

    final_default = dev_winner if (dev_winner != "NONE" and lockbox_verified) else "NOT_VERIFIED"
    print(f"DEV_WINNER={dev_winner}")
    print(f"LOCKBOX_VERIFIED={'YES' if lockbox_verified else 'NO'}")
    print(f"FINAL_DEFAULT={final_default}")

    report = {
        "run_id": run_id,
        "dataset": args.dataset,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "dev_winner": dev_winner,
        "lockbox_verdict": "VERIFIED" if lockbox_verified else "NOT VERIFIED",
        "final_default": final_default,
        "dev_scoreboard": str(out / "dev_scoreboard.csv"),
        "lockbox_scoreboard": str(out / "lockbox_scoreboard.csv"),
        "dev_summary": dev["summary"],
        "lockbox_summary": lock["summary"],
    }
    (out / "improvement_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
