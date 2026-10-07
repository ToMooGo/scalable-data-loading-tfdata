.PHONY: help install lint format test test-all quick train eval monitor monitor-up api up up-quick serve down logs report notebooks clean

PY ?= python

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## local dev environment (Python 3.12+)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

lint:            ## ruff lint + format check
	ruff check . && ruff format --check .

format:          ## auto-format
	ruff check --fix . && ruff format .

test:            ## unit + API tests, offline (~2 min)
	$(PY) -m pytest -m "not integration" -q

test-all:        ## + Prefect/MLflow integration test on the quick config (downloads MNIST once)
	$(PY) -m pytest -q

quick:           ## train -> evaluate -> deploy with the quick config (~5 min, local SQLite MLflow; writes reports/quick/)
	$(PY) run_flow.py --config configs/quick_flow_config.yaml

train:           ## full experiment: regenerates reports/ (~45 minutes on 2 CPUs)
	$(PY) run_flow.py --config configs/full_flow_config.yaml

eval:            ## re-evaluate the newest registered model against the quality gates
	$(PY) run_flow.py --config configs/full_flow_config.yaml --flow eval

monitor:         ## drift check on logged predictions
	$(PY) run_flow.py --config configs/monitor_flow_config.yaml

# absolute paths: the API runs from services/api, the flows from the repo root
# REPORTS_DIR: the Pipeline results tab shows reports/ (the full run); REPORTS_DIR=$(CURDIR)/reports/quick shows the quick run
LOCAL_ENV = MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-sqlite:///$(CURDIR)/mlflow.db} DATABASE_URL=$${DATABASE_URL:-sqlite:///$(CURDIR)/app.db} REPORTS_DIR=$${REPORTS_DIR:-$(CURDIR)/reports} PYTHONPATH=$(CURDIR)/src

api:             ## serve the API + UI locally on :8000 (needs a trained @champion: `make quick`)
	cd services/api && $(LOCAL_ENV) uvicorn app.main:create_app --factory --reload --port 8000

DC = HOST_UID=$$(id -u) HOST_GID=$$(id -g) GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose

up:              ## whole stack in Docker: Postgres, MLflow, Prefect, API, then the pipeline
	$(DC) up --build

up-quick:        ## same, with the quick config
	FLOW_CONFIG=configs/quick_flow_config.yaml $(DC) up --build

serve:           ## restart the stack WITHOUT retraining: the API loads the registered @champion
	$(DC) up -d api prefect

monitor-up:      ## hourly drift check as a Prefect deployment (docker)
	$(DC) --profile monitor up -d monitor

down:            ## stop the stack (add -v to wipe volumes)
	docker compose down

logs:
	docker compose logs -f api pipeline

report:          ## tables from the last full run (reports/) -> LaTeX -> PDF (pdflatex, run twice)
	$(PY) reports/technical_report/make_tables.py
	cd reports/technical_report && pdflatex -interaction=nonstopmode technical_report.tex && pdflatex -interaction=nonstopmode technical_report.tex

notebooks:       ## re-execute the notebooks in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf mlruns mlflow.db app.db .pytest_cache .ruff_cache reports/quick
