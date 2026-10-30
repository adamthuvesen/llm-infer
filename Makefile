.DEFAULT_GOAL := check
.PHONY: help install fmt lint typecheck architecture-check test visualizer-test evidence-check reproduce check

help:  ## List commands and their purpose
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target>\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Sync the dev environment
	uv sync --extra dev --extra serving

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

architecture-check:  ## Check that policy readers do not import model execution
	uv run lint-imports --no-logo

typecheck:  ## Check package and evidence CLI types
	uv run mypy

test:  ## Run unit tests
	uv run python -m pytest

evidence-check:  ## Check committed records, public tables, figure regen, and A/B verdicts
	uv run scripts/check_benchmark_evidence.py

visualizer-test:  ## Test trace loading in Node
	node --test visualizer/trace_loader.test.mjs

reproduce: evidence-check  ## Re-derive every committed verdict from its records (no GPU)

check: architecture-check lint typecheck test evidence-check visualizer-test  ## Run all CPU verification checks
