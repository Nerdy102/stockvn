from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_alpha_v2_smoke_walkforward() -> None:
    c = TestClient(create_app())
    tr = c.post('/ml/train')
    assert tr.status_code == 200
    d = c.post('/ml/diagnostics', json={})
    assert d.status_code == 200
    body = d.json()
    assert body.get('metrics')
    metrics = body['metrics']
    for key in ['rank_ic_mean', 'ic_decay_1', 'ic_decay_21', 'sharpe_lo', 'sharpe_hi']:
        assert key in metrics
        assert metrics[key] == metrics[key]
    bt = c.post('/ml/backtest', json={'mode': 'v2'})
    assert bt.status_code == 200
    b = bt.json()
    assert 'walk_forward' in b
    m_rows = b['walk_forward'].get('metrics', [])
    assert isinstance(m_rows, list) and len(m_rows) > 0
