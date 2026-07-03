.DEFAULT_GOAL := check
.PHONY: install fmt lint test check

install:  ## Sync the dev environment
	uv sync --extra dev

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run unit tests
	uv run python -m pytest

check: lint test  ## The gate: lint + format-check + unit tests
