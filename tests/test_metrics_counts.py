from __future__ import annotations

import datetime as dt
import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_metrics_counts() -> None:
    c = TestClient(app)
    before = c.get('/oms/metrics/simple').json()

    draft = c.post(
        '/oms/draft',
        json={
            'user_id': 'u-metrics',
            'market': 'vn',
            'symbol': 'FPT',
            'side': 'BUY',
            'qty': 100,
            'price': 10000,
            'model_id': 'm1',
            'config_hash': f'cfg-{uuid.uuid4().hex[:6]}',
            'ts_bucket': uuid.uuid4().hex[:8],
        },
    ).json()['order']
    c.post('/oms/approve', json={'order_id': draft['id'], 'confirm_token': draft['confirm_token'], 'checkboxes': {'ok': True}})
    c.post('/oms/execute', json={'order_id': draft['id'], 'data_freshness': {'as_of_date': dt.date.today().isoformat()}, 'portfolio_snapshot': {'cash': 2_000_000_000.0, 'nav_est': 2_000_000_000.0, 'orders_today': 0}})

    after = c.get('/oms/metrics/simple').json()
    assert after['orders_created_total'] >= before['orders_created_total'] + 1
    assert after['orders_executed_total'] >= before['orders_executed_total'] + 1
