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

architecture-check:  ## Check dependency direction and policy isolation
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

# The test suite and the evidence check are the slow parts and share no files, so they run in
# parallel. Evidence and visualizer output is buffered and printed after the tests finish.
check: architecture-check lint typecheck  ## Run all CPU verification checks
	@log=$$(mktemp); \
	$(MAKE) --no-print-directory evidence-check visualizer-test >$$log 2>&1 & side=$$!; \
	$(MAKE) --no-print-directory test; test_status=$$?; \
	wait $$side; side_status=$$?; cat $$log; rm -f $$log; \
	exit $$((test_status || side_status))
