.PHONY: setup run-api run-worker run-ui run-stream-ingestor run-realtime replay-demo verify-program rt-load-test rt-chaos-test rt-verify test lint format docker-up docker-down quality-gate bronze-verify bronze-cleanup replay-smoke

PYTHONPATH := packages/core:packages/data:packages:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps
VENV := .venv
PY := $(VENV)/bin/python
PY_RUNTIME := $(shell [ -x $(PY) ] && echo $(PY) || echo python)
PIP := $(VENV)/bin/pip
SSI_STREAM_MOCK_MESSAGES_PATH ?= tests/fixtures/ssi_streaming

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
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m mypy .

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
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m mypy scripts/ui_guardrail_check.py scripts/dev_reset_db.py scripts/dev_seed_minimal.py scripts/verify_program.py scripts/replay_events.py packages/data/contracts packages/core/observability tools/realtime_harness
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q tests/test_contract_hash_stability.py tests/test_schema_registry_versioning.py tests/test_bronze_to_silver_deterministic.py tests/test_silver_validations_reject_invalid.py tests/test_lineage_preserved.py tests/test_schema_monitor_detects_key_change.py tests/test_ci_forbidden_strings_guardrail.py tests/test_make_targets_exist.py tests/test_rt_harness.py tests/test_gateway_event_id_deterministic.py tests/test_gateway_dedup_skips_duplicates.py tests/test_event_log_rotation_checksum_verify.py tests/test_replay_publishes_ordered.py tests/test_bar_alignment_lunch_split.py tests/test_bar_hash_deterministic.py tests/test_late_event_correction_policy.py tests/test_redis_hot_cache_rolls_last_200.py tests/test_incremental_matches_batch.py tests/test_trend_definition_exact.py tests/test_intraday_cooldown_26_bars.py tests/test_alert_dsl_eval_on_bar_close.py tests/test_signal_idempotent.py
	$(PY_RUNTIME) scripts/quality_gate.py

bronze-verify:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bronze_verify

bronze-cleanup:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bronze_cleanup

replay-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.replay_smoke

rt-load-test:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m tools.realtime_harness.run_load --symbols 500 --days 2 --seed 42 --out artifacts/verification/MEGA09_load_results.json

rt-chaos-test:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m tools.realtime_harness.run_chaos --seed 42 --out artifacts/verification/MEGA09_chaos_results.json

rt-verify: rt-load-test rt-chaos-test

verify-program:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.verify_program
