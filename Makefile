.DEFAULT_GOAL := check
SHELL := bash
.SHELLFLAGS := -euo pipefail -c
# uv run and uv sync fail on a lockfile that no longer matches pyproject.toml instead of relocking.
export UV_LOCKED := 1
.PHONY: help install fix lint typecheck architecture-check test visualizer-deps visualizer-test evidence-check visualizer-lint slow check verify

help:  ## List commands and their purpose
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target>\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Sync the dev environment
	uv sync --extra dev --extra serving
	npm ci --prefix visualizer

fix: visualizer-deps  ## Auto-format and apply safe lint fixes
	uv run ruff check --fix . .agents/skills/verify-llm-infer/verify
	uv run ruff format . .agents/skills/verify-llm-infer/verify
	cd visualizer && ./node_modules/.bin/oxlint --fix --type-aware && ./node_modules/.bin/oxfmt

lint:  ## Lint and format-check (no changes)
	uv run ruff check . .agents/skills/verify-llm-infer/verify
	uv run ruff format --check . .agents/skills/verify-llm-infer/verify

architecture-check:  ## Check dependency direction and policy isolation
	uv run lint-imports --no-logo

typecheck:  ## Check package, scripts, tests, and the verify helper with basedpyright strict
	uv run basedpyright
	uv run basedpyright --project .agents/skills/verify-llm-infer

test:  ## Run unit tests
	uv run python -m pytest

slow:  ## Run the opt-in 3B Qwen reference checks and figure regeneration (needs network for plotly)
	uv run python -m pytest tests/correctness -q -m slow
	uv run scripts/check_benchmark_evidence.py --figures

evidence-check:  ## Check committed records, public tables, and A/B verdicts
	uv run scripts/check_benchmark_evidence.py

verify:  ## Drive the real server, loadgen, and Modal launch paths on CPU (ARGS=all|list|control|<drive>)
	@uv run python .agents/skills/verify-llm-infer/verify $(ARGS)

visualizer-test:  ## Test trace loading in Node
	node --test visualizer/trace_loader.test.mjs

visualizer-deps:
	@test -x visualizer/node_modules/.bin/oxlint && test -x visualizer/node_modules/.bin/oxfmt || { printf '%s\n' 'Visualizer tools are missing. Run make install.' >&2; exit 1; }

visualizer-lint: visualizer-deps  ## Type-check, lint, and format-check the visualizer
	cd visualizer && ./node_modules/.bin/oxlint --type-aware --type-check --deny-warnings && ./node_modules/.bin/oxfmt --check


# Types, tests, and the evidence check are the slow parts and share no files, so they run in
# parallel. Type, evidence, and visualizer output is buffered and printed after the tests finish.
check: visualizer-deps architecture-check lint  ## Run all CPU verification checks
	@log=$$(mktemp); \
	$(MAKE) --no-print-directory -k typecheck evidence-check visualizer-test visualizer-lint >$$log 2>&1 & side=$$!; \
	test_status=0; $(MAKE) --no-print-directory test || test_status=$$?; \
	side_status=0; wait $$side || side_status=$$?; cat $$log; rm -f $$log; \
	exit $$((test_status || side_status))
