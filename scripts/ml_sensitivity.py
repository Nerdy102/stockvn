from __future__ import annotations

from core.ml.backtest import run_sensitivity


if __name__ == "__main__":
    print(run_sensitivity({"Sharpe": 0.5}))
