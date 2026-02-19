from __future__ import annotations

import datetime as dt
import uuid

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_kiosk_draft_confirm_flow_mocked() -> None:
    c = TestClient(app)

    draft = c.post(
        '/oms/draft',
        json={
            'user_id': 'kiosk-user',
            'market': 'vn',
            'symbol': 'FPT',
            'side': 'BUY',
            'qty': 100,
            'price': 10000,
            'model_id': 'model_1',
            'config_hash': 'kiosk-v3',
            'ts_bucket': uuid.uuid4().hex[:8],
            'reason_short': 'Tín hiệu thuận xu hướng.',
        },
    )
    assert draft.status_code == 200
    order = draft.json()['order']

    approve = c.post(
        '/oms/approve',
        json={'order_id': order['id'], 'confirm_token': order['confirm_token'], 'checkboxes': {'risk': True, 'edu': True}},
    )
    assert approve.status_code == 200

    execute = c.post(
        '/oms/execute',
        json={
            'order_id': order['id'],
            'data_freshness': {'as_of_date': dt.date.today().isoformat()},
            'portfolio_snapshot': {'cash': 2_000_000_000.0, 'nav_est': 2_000_000_000.0, 'orders_today': 0},
            'drift_alerts': {'drift_paused': False, 'kill_switch_on': False},
        },
    )
    assert execute.status_code == 200
    assert execute.json()['order']['status'] == 'FILLED'
