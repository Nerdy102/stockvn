import pandas as pd
from core.backtest.report import build_report


def test_backtest_report_has_metrics_and_disclaimer() -> None:
    r = pd.Series([0.01, -0.005, 0.002, 0.003])
    eq = (1 + r).cumprod()
    rep = build_report(eq, r)
    for k in [
        "cagr",
        "max_drawdown",
        "sharpe",
        "sortino",
        "profit_factor",
        "expectancy",
        "disclaimer",
    ]:
        assert k in rep
    assert "quá khứ" in rep["disclaimer"]
