from __future__ import annotations

from api_fastapi.main import create_app
from fastapi.testclient import TestClient


def test_research_v4_checklist_logic_correct() -> None:
    c = TestClient(create_app())
    bt = c.post('/ml/backtest', json={'mode': 'v2', 'tag': 'checklist'})
    assert bt.status_code == 200
    h = bt.json()['run_hash']

    ok = c.get('/ml/research_v4/promotion_checklist', params={'run_hash': h, 'drift_ok': 'true', 'capacity_ok': 'true'})
    assert ok.status_code == 200
    body = ok.json()
    rules = {r['rule']: bool(r['pass']) for r in body.get('rules', [])}
    assert rules.get('drift_ok') is True
    assert rules.get('capacity_ok') is True

    bad = c.get('/ml/research_v4/promotion_checklist', params={'run_hash': h, 'drift_ok': 'false', 'capacity_ok': 'false'})
    assert bad.status_code == 200
    b = bad.json()
    rules_b = {r['rule']: bool(r['pass']) for r in b.get('rules', [])}
    assert rules_b.get('drift_ok') is False
    assert rules_b.get('capacity_ok') is False
    assert 'drift_ok' in b.get('fail_reasons', [])
    assert 'capacity_ok' in b.get('fail_reasons', [])
