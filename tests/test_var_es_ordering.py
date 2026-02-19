from __future__ import annotations

from core.risk_tail.var_es import es_historical, var_historical


def test_var_es_ordering() -> None:
    returns = [-0.08, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02]
    var95 = var_historical(returns, 0.05)
    es95 = es_historical(returns, 0.05)
    var99 = var_historical(returns, 0.01)
    es99 = es_historical(returns, 0.01)
    assert es95 <= var95
    assert es99 <= var99
