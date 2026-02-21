from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def audit_prices(path: Path, out_md: Path | None = None) -> dict[str, object]:
    df = pd.read_csv(path)
    req = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in req if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if "value_vnd" not in df.columns:
        df["value_vnd"] = df["close"] * df["volume"]

    dup = int(df.duplicated(subset=["symbol", "date"]).sum())
    miss_close = float(df["close"].isna().mean())
    miss_vol = float(df["volume"].isna().mean())

    outlier_thr = 0.2
    df = df.sort_values(["symbol", "date"])
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    outliers = int((df["ret1"].abs() > outlier_thr).sum())
    bad_price = int(
        ((df["close"] <= 0) | (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0)).sum()
    )
    bad_vol = int((df["volume"] <= 0).sum())
    gaps = int(df.groupby("symbol")["date"].nunique().min())
    ca_flags = int((df["ret1"].abs() > 0.30).sum())

    reasons = []
    if miss_close >= 0.02 or miss_vol >= 0.02:
        reasons.append("missing>=2%")
    if dup > 0:
        reasons.append("duplicates>0")
    if gaps < 120:
        reasons.append("min_days<120")
    health = "PASS" if not reasons else "FAIL"

    report = {
        "path": str(path),
        "rows": int(len(df)),
        "schema_ok": True,
        "duplicates": dup,
        "missing_close_ratio": miss_close,
        "missing_volume_ratio": miss_vol,
        "outliers_abs_ret_gt_20pct": outliers,
        "non_positive_price": bad_price,
        "non_positive_volume": bad_vol,
        "possible_corporate_actions": ca_flags,
        "min_days_per_symbol": gaps,
        "data_health_score": health,
        "reasons": reasons,
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }

    if out_md is not None:
        lines = [
            "# Data Audit",
            f"- path: `{report['path']}`",
            f"- rows: {report['rows']}",
            f"- date_range: {report['date_min']} -> {report['date_max']}",
            f"- duplicates: {report['duplicates']}",
            f"- missing_close_ratio: {report['missing_close_ratio']:.4f}",
            f"- missing_volume_ratio: {report['missing_volume_ratio']:.4f}",
            f"- outliers_abs_ret_gt_20pct: {report['outliers_abs_ret_gt_20pct']}",
            f"- possible_corporate_actions: {report['possible_corporate_actions']}",
            f"- min_days_per_symbol: {report['min_days_per_symbol']}",
            f"- Data Health Score: **{report['data_health_score']}**",
            f"- reasons: {', '.join(reasons) if reasons else 'none'}",
        ]
        out_md.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-md", default="")
    args = ap.parse_args()
    out = Path(args.output_md) if args.output_md else None
    rep = audit_prices(Path(args.input), out)
    print(rep)
