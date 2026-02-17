from __future__ import annotations

import argparse

import pandas as pd
from core.backtest.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_demo/prices_demo_1d.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["symbol"] != "VNINDEX"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    # Simple walk-forward proxy: split by date halves
    mid = df["date"].quantile(0.5)
    test = df[df["date"] >= mid].copy()
    ret = test.groupby("date")["close"].mean().pct_change().fillna(0.0)
    eq = (1 + ret).cumprod()
    report = build_report(eq, ret)
    print(report)


if __name__ == "__main__":
    main()
