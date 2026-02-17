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
    bt = c.post('/ml/backtest', json={'mode': 'v2'})
    assert bt.status_code == 200
    assert 'walk_forward' in bt.json()
