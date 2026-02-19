from __future__ import annotations

from fastapi.testclient import TestClient

from api_fastapi.main import app
from core.quant_stats.bootstrap import block_bootstrap_ci


def test_bootstrap_bounded() -> None:
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
            'bootstrap_n_iter': 999,
        },
    )
    assert r.status_code == 200
    row = r.json()['leaderboard'][0]
    assert row['ci_net_return']['n_iter'] <= 500

    short = block_bootstrap_ci([0.01, -0.02, 0.03], 'net_return', block_size=10, n_iter=200)
    assert short['ci'] is None
    assert 'ngắn' in short['reason']
