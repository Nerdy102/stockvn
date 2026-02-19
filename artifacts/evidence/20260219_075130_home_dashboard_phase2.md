# Evidence Pack 20260219_075130

## python --version
Python 3.10.19

## git rev-parse HEAD
17654c907017ab0ac777bec49e5baf4e1fd40760

## git status --porcelain
(sạch)

## make verify-offline (trích log)
PYTHONPATH=packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps python -m pytest -q tests/test_simple_mode_models_smoke.py tests/test_order_draft_tick_lot_fee_tax.py tests/test_confirm_execute_paper_updates_ledger.py tests/test_ui_simple_mode_import.py tests/test_api_simple_mode_bounds.py tests/test_age_gating_disclaimer.py tests/test_simple_mode_ui_guardrails.py tests/test_simple_dashboard_payload_smoke.py tests/test_simple_dashboard_bounds.py tests/test_simple_dashboard_determinism_hash.py tests/test_ui_home_dashboard_import.py tests/test_ui_vietnamese_labels_smoke.py
...............                                                          [100%]
15 passed in 6.02s

## make verify-e2e (trích log)
PYTHONPATH=packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps python -m pytest -q tests/test_simple_mode_models_smoke.py tests/test_confirm_execute_paper_updates_ledger.py
..                                                                       [100%]
2 passed in 4.66s

## git status --porcelain (sau cleanup)
(sạch)
