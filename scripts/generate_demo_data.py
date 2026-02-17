from __future__ import annotations

"""Generate synthetic demo CSV data (offline).

This script is OPTIONAL (repo already includes data_demo/).
It exists to demonstrate:
- data is synthetic (no copyright issues)
- tick rounding uses the same market_rules module (no duplicated tick logic)

Run:
  PYTHONPATH=packages/core python -m scripts.generate_demo_data --outdir data_demo
"""

import argparse
import csv
import datetime as dt
import random
from pathlib import Path

from core.market_rules import load_market_rules


def iter_trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += dt.timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data_demo")
    parser.add_argument("--rules", default="configs/market_rules_vn.yaml")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rules = load_market_rules(args.rules)

    tickers = [
        ("FVNA", "FVN Alpha Bank", "HOSE", "Banks", "Commercial Bank", 38500, 2400000000, 5500000, True, False, "KQKD;policy"),
        ("FVNB", "FVN Beta Securities", "HOSE", "Securities", "Brokerage", 22500, 1200000000, 4200000, False, True, "liquidity;cycle"),
    ]
    benchmark = ("VNINDEX", "VN-Index (Synthetic)", "HOSE", "Index", "Benchmark", 1250.0, 0, 0, False, False, "")

    # tickers_demo.csv
    with (outdir / "tickers_demo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","name","exchange","sector","industry","shares_outstanding","market_cap","is_bank","is_broker","tags"])
        for sym, name, ex, sec, ind, base_px, shares, *_ in tickers:
            mcap = float(base_px) * float(shares)
            w.writerow([sym, name, ex, sec, ind, shares, int(mcap), 0, 0, ""])
        w.writerow([benchmark[0], benchmark[1], benchmark[2], benchmark[3], benchmark[4], 0, 0, 0, 0, ""])

    start = dt.date(2025, 3, 3)
    end = dt.date(2026, 2, 13)
    days = iter_trading_days(start, end)
    rng = random.Random(42)

    def gen_daily(symbol: str, base_price: float, base_vol: int):
        close = float(base_price)
        drift = 0.00025
        vol = 0.018 if symbol != "VNINDEX" else 0.010
        for dday in days:
            r = drift + vol * rng.gauss(0.0, 1.0)
            prev_close = close
            close = max(1.0, prev_close * (1.0 + r))
            open_ = prev_close * (1.0 + 0.003 * rng.gauss(0.0, 1.0))
            high = max(open_, close) * (1.0 + abs(0.006 * rng.gauss(0.0, 1.0)))
            low = min(open_, close) * (1.0 - abs(0.006 * rng.gauss(0.0, 1.0)))
            volume = int(max(100, base_vol * (1.0 + 0.35 * rng.gauss(0.0, 1.0)))) if base_vol > 0 else 0

            if symbol != "VNINDEX":
                open_ = rules.round_price(open_, instrument="stock")
                high = rules.round_price(high, instrument="stock", direction="up")
                low = rules.round_price(low, instrument="stock", direction="down")
                close = rules.round_price(close, instrument="stock")
            else:
                open_, high, low, close = [round(x, 2) for x in (open_, high, low, close)]
            yield dday, open_, high, low, close, volume

    with (outdir / "prices_demo_1d.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date","symbol","open","high","low","close","volume","value_vnd"])
        for sym, *_rest in tickers:
            base_px = float([x for x in tickers if x[0] == sym][0][5])
            base_vol = int([x for x in tickers if x[0] == sym][0][7])
            for dday, o, h, l, c, v in gen_daily(sym, base_px, base_vol):
                w.writerow([dday.isoformat(), sym, o, h, l, c, v, int(c * v)])
        for dday, o, h, l, c, v in gen_daily("VNINDEX", float(benchmark[5]), 0):
            w.writerow([dday.isoformat(), "VNINDEX", o, h, l, c, v, 0])

    print(f"Generated demo data to: {outdir}")


if __name__ == "__main__":
    main()
