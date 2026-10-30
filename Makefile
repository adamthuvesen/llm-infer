.DEFAULT_GOAL := check
.PHONY: help install fix lint typecheck architecture-check test visualizer-test evidence-check slow check verify

help:  ## List commands and their purpose
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target>\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Sync the dev environment
	uv sync --extra dev --extra serving

fix:  ## Auto-format and apply safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

architecture-check:  ## Check dependency direction and policy isolation
	uv run lint-imports --no-logo

typecheck:  ## Check package, scripts, tests, and the verify helper with basedpyright strict
	uv run basedpyright
	uv run python scripts/check_basedpyright_ratchet.py

test:  ## Run unit tests
	uv run python -m pytest

slow:  ## Run the opt-in 3B Qwen reference checks (CPU, minutes)
	uv run python -m pytest tests/correctness -q -m slow

evidence-check:  ## Check committed records, public tables, figure regen, and A/B verdicts
	uv run scripts/check_benchmark_evidence.py

verify:  ## Drive the real server, loadgen, and Modal launch paths on CPU (ARGS=all|list|control|<drive>)
	@uv run python .agents/skills/verify-llm-infer/verify $(ARGS)

visualizer-test:  ## Test trace loading in Node
	node --test visualizer/trace_loader.test.mjs


# Types, tests, and the evidence check are the slow parts and share no files, so they run in
# parallel. Type, evidence, and visualizer output is buffered and printed after the tests finish.
check: architecture-check lint  ## Run all CPU verification checks
	@log=$$(mktemp); \
	$(MAKE) --no-print-directory typecheck evidence-check visualizer-test >$$log 2>&1 & side=$$!; \
	$(MAKE) --no-print-directory test; test_status=$$?; \
	wait $$side; side_status=$$?; cat $$log; rm -f $$log; \
	exit $$((test_status || side_status))
