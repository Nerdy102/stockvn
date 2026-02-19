from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_run_compare_includes_advanced_validation() -> None:
    c = TestClient(app)
    r = c.post(
        '/simple/run_compare',
        json={
            'symbols': ['FPT', 'HPG', 'VCB', 'MWG', 'SSI'],
            'timeframe': '1D',
            'lookback_days': 252,
            'detail_level': 'chi tiết',
            'engine_version': 'v3',
            'market': 'vn',
            'trading_type': 'spot_paper',
            'include_equity_curve': False,
            'include_trades': False,
            'execution': 'giá đóng cửa (close)',
            'enable_bootstrap': False,
            'bootstrap_b': 120,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 'cscv' in body
    assert 'rc_spa' in body
    assert 'allocation_suggestion' in body
    row = body['leaderboard'][0]
    assert 'tail_risk' in row
    for k in ['var95', 'es95', 'var99', 'es99']:
        assert k in row['tail_risk']
