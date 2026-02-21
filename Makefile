.PHONY: setup run-api run-worker run-ui run-ui-kiosk run-stream-ingestor run-realtime replay-demo verify-program rt-load-test rt-chaos-test rt-verify test lint format docker-up docker-down quality-gate ui-guardrails bronze-verify bronze-cleanup replay-smoke verify-regression fetch-vn10 bootstrap-vn10 eval-vn10 eval-chat model-chat redis rt-btc-ingest rt-btc-bars rt-btc-signals rt-btc-demo-order

PYTHONPATH := .:services:packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps
VENV := .venv
PY := $(VENV)/bin/python
PY_RUNTIME := $(shell [ -x $(PY) ] && echo $(PY) || echo python)
PIP := $(VENV)/bin/pip
SSI_STREAM_MOCK_MESSAGES_PATH ?= tests/fixtures/ssi_streaming
REDIS_URL ?= redis://localhost:6379/0

setup:
	python -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	@if [ ! -f .env ]; then cp .env.example .env; fi
	PYTHONPATH=$(PYTHONPATH) $(PY) -m scripts.seed_demo_data

run-api:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m uvicorn api_fastapi.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m worker_scheduler.main --interval-minutes 15

run-ui:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m streamlit run apps/dashboard_streamlit/app.py --server.port 8501

run-ui-kiosk:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m streamlit run apps/web_kiosk/app.py --server.port 8502

run-stream-ingestor:
	SSI_STREAM_MOCK_MESSAGES_PATH=$(SSI_STREAM_MOCK_MESSAGES_PATH) PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m stream_ingestor.main

run-realtime:
	docker compose -f infra/docker-compose.yml --env-file .env --profile realtime up --build

replay-demo:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.replay_events --fixture tests/fixtures/replay/event_log_fixture.jsonl --speed max

test:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q

lint:
	$(PY_RUNTIME) -m ruff check .
	$(PY_RUNTIME) -m black --check .
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m mypy --explicit-package-bases .

format:
	$(PY_RUNTIME) -m black .
	$(PY_RUNTIME) -m ruff check . --fix

docker-up:
	docker compose -f infra/docker-compose.yml --env-file .env up --build

docker-down:
	docker compose -f infra/docker-compose.yml down -v

quality-gate:
	$(PY_RUNTIME) -m ruff check scripts/ui_guardrail_check.py scripts/dev_reset_db.py scripts/dev_seed_minimal.py scripts/verify_program.py scripts/replay_events.py packages/data/contracts packages/core/observability tools/realtime_harness tests/test_contract_hash_stability.py tests/test_schema_registry_versioning.py tests/test_bronze_to_silver_deterministic.py tests/test_silver_validations_reject_invalid.py tests/test_lineage_preserved.py tests/test_schema_monitor_detects_key_change.py tests/test_ci_forbidden_strings_guardrail.py tests/test_make_targets_exist.py tests/test_rt_harness.py tests/test_gateway_event_id_deterministic.py tests/test_gateway_dedup_skips_duplicates.py tests/test_event_log_rotation_checksum_verify.py tests/test_replay_publishes_ordered.py tests/test_bar_alignment_lunch_split.py tests/test_bar_hash_deterministic.py tests/test_late_event_correction_policy.py tests/test_redis_hot_cache_rolls_last_200.py tests/test_incremental_matches_batch.py tests/test_trend_definition_exact.py tests/test_intraday_cooldown_26_bars.py tests/test_alert_dsl_eval_on_bar_close.py tests/test_signal_idempotent.py
	$(PY_RUNTIME) -m black --check scripts/ui_guardrail_check.py scripts/dev_reset_db.py scripts/dev_seed_minimal.py scripts/verify_program.py scripts/replay_events.py packages/data/contracts packages/core/observability tools/realtime_harness tests/test_contract_hash_stability.py tests/test_schema_registry_versioning.py tests/test_bronze_to_silver_deterministic.py tests/test_silver_validations_reject_invalid.py tests/test_lineage_preserved.py tests/test_schema_monitor_detects_key_change.py tests/test_ci_forbidden_strings_guardrail.py tests/test_make_targets_exist.py tests/test_rt_harness.py tests/test_gateway_event_id_deterministic.py tests/test_gateway_dedup_skips_duplicates.py tests/test_event_log_rotation_checksum_verify.py tests/test_replay_publishes_ordered.py tests/test_bar_alignment_lunch_split.py tests/test_bar_hash_deterministic.py tests/test_late_event_correction_policy.py tests/test_redis_hot_cache_rolls_last_200.py tests/test_incremental_matches_batch.py tests/test_trend_definition_exact.py tests/test_intraday_cooldown_26_bars.py tests/test_alert_dsl_eval_on_bar_close.py tests/test_signal_idempotent.py
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m mypy --explicit-package-bases scripts/ui_guardrail_check.py scripts/dev_reset_db.py scripts/dev_seed_minimal.py scripts/verify_program.py scripts/replay_events.py packages/data/contracts packages/core/observability tools/realtime_harness/chaos_controller.py tools/realtime_harness/generate_synthetic_events.py tools/realtime_harness/replay_to_redis.py tools/realtime_harness/run_chaos.py tools/realtime_harness/run_load.py tools/realtime_harness/verify_invariants.py
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q tests/test_contract_hash_stability.py tests/test_schema_registry_versioning.py tests/test_bronze_to_silver_deterministic.py tests/test_silver_validations_reject_invalid.py tests/test_lineage_preserved.py tests/test_schema_monitor_detects_key_change.py tests/test_ci_forbidden_strings_guardrail.py tests/test_make_targets_exist.py tests/test_rt_harness.py tests/test_gateway_event_id_deterministic.py tests/test_gateway_dedup_skips_duplicates.py tests/test_event_log_rotation_checksum_verify.py tests/test_replay_publishes_ordered.py tests/test_bar_alignment_lunch_split.py tests/test_bar_hash_deterministic.py tests/test_late_event_correction_policy.py tests/test_redis_hot_cache_rolls_last_200.py tests/test_incremental_matches_batch.py tests/test_trend_definition_exact.py tests/test_intraday_cooldown_26_bars.py tests/test_alert_dsl_eval_on_bar_close.py tests/test_signal_idempotent.py
	$(PY_RUNTIME) scripts/quality_gate.py

ui-guardrails:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.ui_guardrail_check

bronze-verify:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bronze_verify

bronze-cleanup:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bronze_cleanup

replay-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.replay_smoke

rt-load-test:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m tools.realtime_harness.run_load --symbols 500 --days 2 --seed 42 --dry-run --out artifacts/verification/RT_LOAD_REPORT.json

rt-chaos-test:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m tools.realtime_harness.run_chaos --seed 42 --dry-run --out artifacts/verification/RT_CHAOS_REPORT.json

rt-verify: rt-load-test rt-chaos-test
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m tools.realtime_harness.verify_invariants --events artifacts/verification/RT_LOAD_EVENTS.jsonl --out artifacts/verification/RT_VERIFY_REPORT.json

verify-program:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.verify_program


verify-offline:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q tests/test_simple_mode_models_smoke.py tests/test_order_draft_tick_lot_fee_tax.py tests/test_confirm_execute_paper_updates_ledger.py tests/test_ui_simple_mode_import.py tests/test_api_simple_mode_bounds.py tests/test_age_gating_disclaimer.py tests/test_simple_mode_ui_guardrails.py tests/test_simple_dashboard_payload_smoke.py tests/test_simple_dashboard_bounds.py tests/test_simple_dashboard_determinism_hash.py tests/test_ui_home_dashboard_import.py tests/test_ui_vietnamese_labels_smoke.py tests/test_api_kiosk_bounds.py tests/test_api_kiosk_smoke.py tests/test_ui_kiosk_import.py tests/test_ui_kiosk_vietnamese_text_smoke.py tests/test_briefing_output_vi.py tests/test_signal_reason_short_present.py tests/test_run_compare_story_fields.py tests/test_confirm_idempotent_double_click.py tests/test_risk_limits_block_reason_codes.py tests/test_off_session_forces_draft_only.py tests/test_data_quality_gates.py tests/test_no_lookahead_breakout.py tests/test_confidence_bucket_rules.py tests/test_model_rules_smoke.py tests/test_backtest_v2_long_only_math.py tests/test_backtest_v2_short_math.py tests/test_fee_tax_slippage_applied.py tests/test_tick_lot_rounding_vn.py tests/test_determinism_report_id.py tests/test_position_sizing_board_lot.py tests/test_kill_switch_daily_loss.py tests/test_cooldown_blocks_signal.py tests/test_walk_forward_stability_score.py tests/test_drift_slippage_anomaly.py tests/test_gates_fail_returns_neutral.py tests/test_breakout_uses_shift1.py tests/test_confidence_score_bucket_thresholds.py tests/test_signal_audit_written.py tests/test_risk_tags_max_2.py tests/test_reason_short_length.py tests/test_event_driven_no_lookahead_structure.py tests/test_limit_fill_touch_rule.py tests/test_partial_fill_participation.py tests/test_fee_tax_slippage_applied_per_fill.py tests/test_crypto_funding_sign.py tests/test_vn_no_short_enforced.py tests/test_report_determinism_hash.py tests/test_walk_forward_folds_deterministic.py tests/test_stress_suite_scenarios_fixed.py tests/test_readiness_report_hashing.py tests/test_drift_pause_conditions.py tests/test_execute_blocked_on_high_drift.py

verify-e2e:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q tests/test_simple_mode_models_smoke.py tests/test_confirm_execute_paper_updates_ledger.py

verify-live-sandbox:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q tests/test_broker_sandbox_e2e.py

verify-regression:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q tests/test_regression_offline_e2e.py

fetch-vn10:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.fetch_eodhd_vn_universe10

bootstrap-vn10:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bootstrap_vn10_and_seed --reset-db

eval-vn10:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m research.evaluate_model_vn10

eval-chat:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) scripts/print_eval_chat.py

model-chat:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) scripts/print_model_snapshot_chat.py

redis:
	docker run --name stockvn-redis -p 6379:6379 -d redis:7-alpine || docker start stockvn-redis

rt-btc-ingest:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m services.stream_ingestor.binance_trade_ingestor --symbols BTCUSDT --redis $(REDIS_URL)

rt-btc-bars:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m services.bar_builder.run --redis $(REDIS_URL) --exchange CRYPTO

rt-btc-signals:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m services.realtime_signal_engine.run --redis $(REDIS_URL)

rt-btc-demo-order:
	PYTHONPATH=$(PYTHONPATH) REDIS_URL=$(REDIS_URL) $(PY_RUNTIME) -m scripts.demo_crypto_oms_realtime_exec
