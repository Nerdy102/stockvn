.PHONY: setup run-api run-worker run-ui test lint format docker-up docker-down quality-gate

PYTHONPATH := packages/core:packages/data:services/api_fastapi:services/worker_scheduler:apps
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	@if [ ! -f .env ]; then cp .env.example .env; fi
	PYTHONPATH=$(PYTHONPATH) $(PY) -m scripts.seed_demo_data

run-api:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m uvicorn api_fastapi.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m worker_scheduler.main --interval-minutes 15

run-ui:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m streamlit run apps/dashboard_streamlit/app.py --server.port 8501

test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .
	PYTHONPATH=$(PYTHONPATH) $(PY) -m mypy .

format:
	$(PY) -m black .
	$(PY) -m ruff check . --fix

docker-up:
	docker compose -f infra/docker-compose.yml --env-file .env up --build

docker-down:
	docker compose -f infra/docker-compose.yml down -v

quality-gate:
	$(PY) -m ruff check .
	$(PY) -m black --check .
	PYTHONPATH=$(PYTHONPATH) $(PY) -m mypy .
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest -q
	$(PY) scripts/quality_gate.py
