from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app


def test_run_compare_includes_stat_fields() -> None:
    c = TestClient(app)
    r = c.post(
        '/simple/run_compare',
        json={
            'symbols': ['FPT', 'HPG', 'VCB'],
            'timeframe': '1D',
            'lookback_days': 252,
            'detail_level': 'chi tiết',
            'engine_version': 'v3',
            'market': 'vn',
            'trading_type': 'spot_paper',
            'execution': 'giá đóng cửa (close)',
            'enable_bootstrap': True,
            'bootstrap_n_iter': 150,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 'multiple_testing_disclosure' in body
    row = body['leaderboard'][0]
    for k in ['dsr', 'psr_0', 'psr_05', 'stats_confidence_bucket', 'multiple_testing_disclosure']:
        assert k in row
    assert row['stats_confidence_bucket'] in {'Cao', 'Vừa', 'Thấp'}
