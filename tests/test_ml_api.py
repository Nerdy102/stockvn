from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_ml_models_endpoint() -> None:
    c = TestClient(create_app())
    r = c.get('/ml/models')
    assert r.status_code == 200
    assert 'ensemble_v2' in r.json() and 'ensemble_v1' in r.json()


def test_ml_train_backtest_dev() -> None:
    c = TestClient(create_app())
    r1 = c.post('/ml/train')
    assert r1.status_code == 200
    r2 = c.post('/ml/backtest', json={})
    assert r2.status_code == 200
    assert 'disclaimer' in r2.json()
