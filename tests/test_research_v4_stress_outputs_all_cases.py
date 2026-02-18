from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


EXPECTED = {'cost_x2', 'fill_x0_5', 'remove_best5', 'base_bps_plus10'}


def test_research_v4_stress_outputs_all_cases() -> None:
    c = TestClient(create_app())
    bt = c.post('/ml/backtest', json={'mode': 'v2', 'tag': 'stress'})
    assert bt.status_code == 200
    h = bt.json()['run_hash']

    r = c.get('/ml/research_v4/stress', params={'run_hash': h})
    assert r.status_code == 200
    cases = set((r.json().get('cases') or {}).keys())
    assert EXPECTED.issubset(cases)
