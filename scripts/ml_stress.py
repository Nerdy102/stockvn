from __future__ import annotations

from core.ml.backtest import run_stress


if __name__ == "__main__":
    print(run_stress({"CAGR": 0.1, "MDD": -0.2, "Sharpe": 0.7}))
