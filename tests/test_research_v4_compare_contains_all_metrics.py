from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


FIXED = {"CAGR", "Sharpe", "MDD", "turnover", "costs", "PSR", "DSR", "PBO", "RC", "SPA"}


def test_research_v4_compare_contains_all_metrics() -> None:
    c = TestClient(create_app())
    r1 = c.post('/ml/backtest', json={'mode': 'v2', 'tag': 'a'})
    r2 = c.post('/ml/backtest', json={'mode': 'v2', 'tag': 'b'})
    assert r1.status_code == 200 and r2.status_code == 200
    h1 = r1.json()['run_hash']
    h2 = r2.json()['run_hash']

    cmp = c.get('/ml/research_v4/compare', params={'run_a': h1, 'run_b': h2})
    assert cmp.status_code == 200
    rows = cmp.json().get('metrics', [])
    names = {str(r.get('metric')) for r in rows}
    assert FIXED.issubset(names)
