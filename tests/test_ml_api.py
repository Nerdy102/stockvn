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


def test_ml_predict_fields_v2() -> None:
    c = TestClient(create_app())
    c.post('/ml/train')
    r = c.get('/ml/predict', params={'date': '05-10-2024', 'universe': 'ALL'})
    assert r.status_code == 200
    rows = r.json()
    if rows:
        for k in ['symbol', 'date', 'score_final', 'mu', 'uncert', 'score_rank_z']:
            assert k in rows[0]
