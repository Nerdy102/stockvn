.PHONY: setup run-api run-worker run-ui run-stream-ingestor test lint format docker-up docker-down quality-gate bronze-verify bronze-cleanup

PYTHONPATH := packages/core:packages/data:services/api_fastapi:services/worker_scheduler:services/stream_ingestor:apps
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
	$(PY_RUNTIME) -m ruff check .
	$(PY_RUNTIME) -m black --check .
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m mypy .
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m pytest -q
	$(PY_RUNTIME) scripts/quality_gate.py

bronze-verify:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bronze_verify

bronze-cleanup:
	PYTHONPATH=$(PYTHONPATH) $(PY_RUNTIME) -m scripts.bronze_cleanup
