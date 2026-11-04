.PHONY: help install up down logs migrate revision lint fmt types test test-contract check dev smoke audition

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## Create the venv on python 3.11 and install dev dependencies
	uv venv --python 3.11
	uv pip install -e ".[dev]"

up:  ## Start postgres and redis
	docker compose up -d postgres redis

down:  ## Stop everything
	docker compose down

logs:  ## Tail the api container
	docker compose logs -f api

migrate:  ## Apply migrations
	.venv/bin/alembic upgrade head

revision:  ## Autogenerate a migration: make revision m="message"
	.venv/bin/alembic revision --autogenerate -m "$(m)"

dev:  ## Run the api on the host with reload
	.venv/bin/uvicorn app.main:app --reload --port 8000

lint:  ## Lint
	.venv/bin/ruff check app tests scripts

fmt:  ## Format
	.venv/bin/ruff format app tests scripts
	.venv/bin/ruff check --fix app tests scripts

types:  ## Type-check
	.venv/bin/mypy app

test:  ## Unit + integration tests, no network
	.venv/bin/pytest

test-contract:  ## Adapter tests against recorded payloads, still no network
	.venv/bin/pytest -m contract

check: lint types test test-contract  ## Everything that gates a phase

smoke:  ## Hit every real provider once end to end; prints latency and credits spent
	.venv/bin/python scripts/smoke.py

audition:  ## Synthesise one line in every Timbre v3.0 voice so you can pick Dadi's
	.venv/bin/python scripts/audition_voices.py
