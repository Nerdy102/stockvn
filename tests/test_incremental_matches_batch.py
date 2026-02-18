from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))

from core.indicators import add_indicators
from indicators.incremental import IndicatorState, update_indicators_state


def test_incremental_matches_batch() -> None:
    rows = []
    for line in (
        Path("tests/fixtures/market_events_fixture.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        r = json.loads(line)
        rows.append(
            {
                "date": r["provider_ts"],
                "open": r["price"],
                "high": r["price"],
                "low": r["price"],
                "close": r["price"],
                "volume": r["qty"],
            }
        )
    df = pd.DataFrame(rows)
    batch = add_indicators(df)

    st = IndicatorState()
    inc_rows = []
    for _, row in df.iterrows():
        ind, st = update_indicators_state(
            st,
            end_ts=dt.datetime.fromisoformat(str(row["date"]).replace("Z", "+00:00")),
            open_=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        inc_rows.append(ind)
    inc = pd.DataFrame(inc_rows)

    eps = {"EMA20": 1e-6, "MACD": 1e-6, "ATR14": 1e-6, "VWAP": 1e-6, "RSI14": 1e-4}
    diffs = {
        k: float((inc[k] - batch[k].fillna(0.0)).abs().max())
        for k in ["EMA20", "MACD", "ATR14", "VWAP", "RSI14"]
    }
    expected = json.loads(
        Path("tests/golden/incremental_vs_batch_expected.json").read_text(encoding="utf-8")
    )
    assert set(expected["diffs"].keys()) == set(diffs.keys())
    for k, tol in eps.items():
        assert diffs[k] <= tol
        assert abs(diffs[k] - float(expected["diffs"][k])) <= tol
