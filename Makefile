.DEFAULT_GOAL := check
.PHONY: install fmt lint typecheck test evidence-check check

install:  ## Sync the dev environment
	uv sync --extra dev --extra serving

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Check package types
	uv run mypy llm_infer

test:  ## Run unit tests
	uv run python -m pytest

evidence-check:  ## Check committed benchmark evidence and figure regeneration
	uv run scripts/check_benchmark_evidence.py

check: lint typecheck test  ## The gate: lint + format-check + types + unit tests
